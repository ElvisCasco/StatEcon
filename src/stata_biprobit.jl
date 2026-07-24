import Optim

"""
    stata_biprobit(df, depvars, regs; integration_pts=24, level=0.95,
                   quiet=false)
        -> NamedTuple

Stata `biprobit <y1> <y2> <regs>, nolog`. Bivariate probit MLE with
both equations sharing the same regressor list (the symmetric form
Stata uses when there's no `=` between the two equation specs).

Model: y1*ᵢ = X_iβ_1 + ε_{1i},  y_{1i} = 1{y1*ᵢ > 0}; similarly for
y2 with β_2; (ε_1, ε_2) ~ N(0, [[1, ρ], [ρ, 1]]). Joint likelihood
uses the equi-correlation decomposition

    ε_1 = √|ρ| Z + √(1-|ρ|) η_1
    ε_2 = sign(ρ)·√|ρ| Z + √(1-|ρ|) η_2

with Z, η_1, η_2 iid N(0,1). Conditioning on Z makes the joint
probability a 1-D integral evaluated by `integration_pts`-point
Gauss–Hermite quadrature.

For numerical stability ρ is parametrised as `tanh(θ_ρ)` (so the
optimiser walks an unbounded line in θ_ρ). SEs for ρ are
delta-method-transformed back to the (−1, 1) scale.

Returns `(; β1, β2, ρ, V, se_β1, se_β2, se_ρ, ll, n, depvars, regs,
nparam, integration_pts)`.
"""
function stata_biprobit(df::DataFrames.AbstractDataFrame,
                       depvars::AbstractVector{Symbol},
                       regs::AbstractVector{Symbol};
                       regs2::Union{Nothing,AbstractVector{Symbol}} = nothing,
                       integration_pts::Int = 24,
                       level::Float64 = 0.95,
                       quiet::Bool = false)
    length(depvars) == 2 ||
        error("biprobit needs exactly 2 dependent variables")
    regs1 = collect(regs)
    regs2v = regs2 === nothing ? regs1 : collect(regs2)
    cols = unique(vcat(collect(depvars), regs1, regs2v))
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c15_raw.(col))
        end
    end

    y1 = Float64.(_c15_raw.(dfc[!, depvars[1]]))
    y2 = Float64.(_c15_raw.(dfc[!, depvars[2]]))
    X1 = hcat(ones(length(y1)),
              [Float64.(_c15_raw.(dfc[!, r])) for r in regs1]...)
    X2 = hcat(ones(length(y1)),
              [Float64.(_c15_raw.(dfc[!, r])) for r in regs2v]...)
    N = length(y1); k1 = size(X1, 2); k2 = size(X2, 2)
    nparam = k1 + k2 + 1
    coefnames1 = ["_cons", string.(regs1)...]
    coefnames2 = ["_cons", string.(regs2v)...]

    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:(n_pts - 1)]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        return F.values, sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
    end
    gh_x, gh_w = _gauss_hermite(integration_pts)
    z_norm = sqrt(2.0) .* gh_x
    w_norm = gh_w ./ sqrt(Base.pi)

    Φ_(x) = 0.5 * (1 + _c15_erf(x / sqrt(2.0)))

    q1 = @. 2y1 - 1     # ±1 sign vectors
    q2 = @. 2y2 - 1

    function negll(θ)
        β1 = view(θ, 1:k1)
        β2 = view(θ, (k1 + 1):(k1 + k2))
        ρ  = tanh(θ[end])
        absρ = abs(ρ); s_ρ = sign(ρ)
        s = sqrt(max(1 - absρ, eps(Float64)))      # √(1 − |ρ|)
        a = sqrt(absρ)                              # √|ρ|
        Xβ1 = X1 * β1
        Xβ2 = X2 * β2
        ll = zero(eltype(θ))
        for i in 1:N
            P_i = zero(eltype(θ))
            for g in eachindex(z_norm)
                z = z_norm[g]
                u1 = (Xβ1[i] + a * z)            / s
                u2 = (Xβ2[i] + s_ρ * a * z)      / s
                P_i += w_norm[g] *
                       Φ_(q1[i] * u1) * Φ_(q2[i] * u2)
            end
            ll += log(max(P_i, eps(Float64)))
        end
        return -ll
    end

    # Warm start: β = 0, θ_ρ = atanh(0.3) → ρ ≈ 0.291. Must be NON-zero
    # because the equi-correlation decomposition uses `sign(ρ)` and
    # `√|ρ|`, which kink at ρ = 0 (ForwardDiff returns a zero gradient
    # there and LBFGS terminates immediately).
    θ0 = zeros(nparam); θ0[end] = atanh(0.3)
    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 4000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    ll = -negll(θ̂)                                 # force Float64

    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x); xpi[i] += h_[i]; xmi[i] -= h_[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h_[i]^2
            for j in (i+1):nθ
                xpp = copy(x); xpp[i] += h_[i]; xpp[j] += h_[j]
                xpm = copy(x); xpm[i] += h_[i]; xpm[j] -= h_[j]
                xmp = copy(x); xmp[i] -= h_[i]; xmp[j] += h_[j]
                xmm = copy(x); xmm[i] -= h_[i]; xmm[j] -= h_[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h_[i]*h_[j])
            end
        end
        return H
    end
    V_raw = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂)))

    # Delta-method on ρ = tanh(θ_ρ): dρ/dθ_ρ = 1 - ρ².
    ρ̂ = tanh(θ̂[end])
    Jmat = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    Jmat[end, end] = 1 - ρ̂^2
    V = Jmat * V_raw * Jmat'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β1 = se[1:k1]; se_β2 = se[(k1 + 1):(k1 + k2)]; se_ρ = se[end]
    β1̂ = θ̂[1:k1]; β2̂ = θ̂[(k1 + 1):(k1 + k2)]

    if !quiet
        function g9(x; w::Int = 10, sig::Int = 7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su = sig; s = Printf.@sprintf("%.*g", su, x)
            cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
            while length(s) > cap && su > 1
                su -= 1; s = Printf.@sprintf("%.*g", su, x)
            end
            0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
            lpad(s, w)
        end
        commafmt(num) = begin
            s = string(abs(num)); parts = String[]; i = length(s)
            while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
            (num < 0 ? "-" : "") * join(reverse(parts), ",")
        end
        crit = Distributions.quantile(Distributions.Normal(),
                                      1 - (1 - level) / 2)

        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Bivariate probit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6d\n", "",
                       "Integration pts.", integration_pts)
        Printf.@printf("Log likelihood = %.4f\n\n", ll)

        function _print(label, b, s)
            z = b / s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit*s; hi = b + crit*s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)

        # Stata's biprobit prints slopes first then _cons within each eq.
        ord1 = vcat(2:k1, [1])
        ord2 = vcat(2:k2, [1])
        Printf.@printf("%-12s |\n", string(depvars[1]))
        for r in ord1
            label = r == 1 ? "_cons" : coefnames1[r]
            _print(label, β1̂[r], se_β1[r])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", string(depvars[2]))
        for r in ord2
            label = r == 1 ? "_cons" : coefnames2[r]
            _print(label, β2̂[r], se_β2[r])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", "/athrho")
        _print("athrho", θ̂[end], sqrt(V_raw[end, end]))
        Printf.@printf("%-12s |\n", "/rho")
        _print("rho", ρ̂, se_ρ)
        println("-"^78)
        # LR test of independence (ρ = 0).
        χ2_ind = (ρ̂ / se_ρ)^2
        p_ind  = 1 - Distributions.cdf(Distributions.Chisq(1), χ2_ind)
        Printf.@printf("LR test of ρ = 0: chi2(1) = %.2f   Prob > chi2 = %.4f\n",
                       χ2_ind, p_ind)
    end

    return (; β1 = β1̂, β2 = β2̂, ρ = ρ̂, V, se_β1, se_β2, se_ρ,
              ll, n = N, depvars = collect(depvars),
              regs1 = regs1, regs2 = regs2v,
              # Back-compat alias: when both equations share the same
              # regressor list, `regs` is the obvious single field.
              regs = regs1 == regs2v ? regs1 : regs1,
              nparam, integration_pts)
end

"""
    _biprobit_probs(res, df; integration_pts=24) -> NamedTuple

Stata `predict, pmarg1`/`pmarg2`/`p11`/`p10`/`p01`/`p00` after
`biprobit`. Returns the two marginals and the four joint probabilities
for every row of `df` (rows with missing values in `res.regs` are
skipped — column entries are `missing` there). Uses the same equi-
correlation 1-D Gauss–Hermite quadrature as `stata_biprobit`.

Output NamedTuple fields:
* `pmarg1` = Φ(X β_1)
* `pmarg2` = Φ(X β_2)
* `p11`, `p10`, `p01`, `p00` = Φ_2(±X β_1, ±X β_2; ±ρ) (the four
  cells of the 2×2 joint probability table).
"""
function _biprobit_probs(res, df::DataFrames.AbstractDataFrame;
                        integration_pts::Int = 24)
    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:(n_pts - 1)]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        return F.values, sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
    end
    gh_x, gh_w = _gauss_hermite(integration_pts)
    z_norm = sqrt(2.0) .* gh_x
    w_norm = gh_w ./ sqrt(Base.pi)
    Φ_(x) = 0.5 * (1 + _c15_erf(x / sqrt(2.0)))

    ρ = res.ρ; absρ = abs(ρ); s_ρ = sign(ρ)
    s  = sqrt(max(1 - absρ, eps(Float64)))
    a  = sqrt(absρ)
    β1 = res.β1; β2 = res.β2

    function _bvn(q1::Real, q2::Real, h::Real, k::Real)
        P = 0.0
        for g in eachindex(z_norm)
            z = z_norm[g]
            u1 = (h + a * z) / s
            u2 = (k + s_ρ * a * z) / s
            P += w_norm[g] * Φ_(q1 * u1) * Φ_(q2 * u2)
        end
        return P
    end

    N = DataFrames.nrow(df)
    pmarg1 = Vector{Union{Float64,Missing}}(missing, N)
    pmarg2 = Vector{Union{Float64,Missing}}(missing, N)
    p11    = Vector{Union{Float64,Missing}}(missing, N)
    p10    = Vector{Union{Float64,Missing}}(missing, N)
    p01    = Vector{Union{Float64,Missing}}(missing, N)
    p00    = Vector{Union{Float64,Missing}}(missing, N)

    # Support both shared-regressor (`res.regs`) and per-equation
    # (`res.regs1`, `res.regs2`) results.
    regs1 = hasproperty(res, :regs1) ? res.regs1 : res.regs
    regs2 = hasproperty(res, :regs2) ? res.regs2 : res.regs

    function _xrow(i, rs)
        any_missing = false
        x_vals = Float64[1.0]
        for r in rs
            v = df[i, r]
            if ismissing(v); any_missing = true; break; end
            push!(x_vals, Float64(_c15_raw(v)))
        end
        return any_missing ? nothing : x_vals
    end

    for i in 1:N
        x1 = _xrow(i, regs1); x1 === nothing && continue
        x2 = _xrow(i, regs2); x2 === nothing && continue
        h = LinearAlgebra.dot(x1, β1)
        k = LinearAlgebra.dot(x2, β2)
        pmarg1[i] = Φ_(h)
        pmarg2[i] = Φ_(k)
        p11[i] = _bvn( 1.0,  1.0,  h,  k)
        p10[i] = _bvn( 1.0, -1.0,  h,  k)
        p01[i] = _bvn(-1.0,  1.0,  h,  k)
        p00[i] = _bvn(-1.0, -1.0,  h,  k)
    end
    return (; pmarg1, pmarg2, p11, p10, p01, p00)
end
