import Optim

"""
    stata_mixlogit(df_long; case_var, depvar, fixed_regs, random_reg,
                   integration_pts=20, level=0.95, quiet=false)
        -> NamedTuple

Stata `mixlogit <depvar> <fixed_regs>, group(<case>) rand(<random_reg>)`.
Mixed (random-parameters) logit on long-form data. One regressor's
coefficient is random, normally distributed across cases:

    β_{random,i} ~ N(μ, σ²)

with everything else fixed. Per-case choice probability:

    P(i) = ∫ [exp(β·z_ij* + Vᵢⱼ*) / Σ_l exp(β·z_il + V_il)] φ(β; μ, σ²) dβ

evaluated by `integration_pts`-point Gauss–Hermite quadrature with
substitution β = μ + σ·√2·x_g. ML by LBFGS (autodiff gradient);
σ is parameterised as `exp(lnσ)` so it stays positive. OIM via FD
Hessian; delta-method on lnσ → σ for the SE row.

`fixed_regs` is a `Vector{Symbol}` (may be empty); `random_reg` is a
single `Symbol`. The mixed-logit MLE can land in flat regions — we
wrap the optimiser in a try/catch and surface a warning + the warm-
start value, instead of bubbling up an exception.

Returns `(; β_fixed, μ, σ, θ̂, V, se, ll, n_obs, n_cases,
fixed_regs, random_reg, integration_pts, converged)`.
"""
function stata_mixlogit(df_long::DataFrames.AbstractDataFrame;
                        case_var::Symbol,
                        depvar::Symbol,
                        fixed_regs::AbstractVector{Symbol} = Symbol[],
                        random_reg::Symbol,
                        integration_pts::Int = 20,
                        level::Float64 = 0.95,
                        quiet::Bool = false)
    cs = df_long[!, case_var]
    y  = [Float64(_c15_raw(v)) for v in df_long[!, depvar]]
    Zf = [[Float64(_c15_raw(v)) for v in df_long[!, r]] for r in fixed_regs]
    Zr = [Float64(_c15_raw(v)) for v in df_long[!, random_reg]]
    N  = length(y)
    p_fixed = length(fixed_regs)
    nparam  = p_fixed + 2                      # μ, lnσ

    case_ids = unique(cs); G = length(case_ids)
    grp_rows = Dict{eltype(case_ids), Vector{Int}}()
    for i in 1:N; push!(get!(() -> Int[], grp_rows, cs[i]), i); end

    # Gauss–Hermite (∫ f(x) e^(−x²) dx) via Golub–Welsch.
    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:(n_pts - 1)]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        return F.values, sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
    end
    gh_x, gh_w = _gauss_hermite(integration_pts)
    w_norm = gh_w ./ sqrt(Base.pi)             # ∫ f(β) N(β; μ, σ²) dβ weights

    function deterministic_V(θ, rowidx)
        # V_ij = β_fixed' · z_ij  (does NOT include the random regressor)
        u = zero(eltype(θ))
        for c in 1:p_fixed; u += θ[c] * Zf[c][rowidx]; end
        return u
    end

    function negll(θ)
        T = eltype(θ)
        μ_r  = θ[p_fixed + 1]
        lnσ  = θ[p_fixed + 2]
        σ_r  = exp(lnσ)
        β_g  = [μ_r + σ_r * sqrt(2.0) * x for x in gh_x]   # length G_pts

        ll = zero(T)
        for (_, ridx) in grp_rows
            # Per-case row indices; identify chosen.
            chosen = findfirst(==(1.0), [y[i] for i in ridx])
            chosen === nothing && continue
            chosen_row = ridx[chosen]
            V = [deterministic_V(θ, i) for i in ridx]
            zr = [Zr[i] for i in ridx]
            zr_chosen = zr[chosen]
            v_chosen  = V[chosen]
            # Probability at each quadrature node, then weight.
            P_i = zero(T)
            for g in eachindex(β_g)
                βv = β_g[g]
                # Stable softmax: max(η) - subtract
                η = [V[k] + βv * zr[k] for k in eachindex(V)]
                mη = maximum(η)
                den = sum(exp(η[k] - mη) for k in eachindex(η))
                p_g = exp(v_chosen + βv * zr_chosen - mη) / den
                P_i += w_norm[g] * p_g
            end
            ll += log(max(P_i, eps(Float64)))
        end
        return -ll
    end

    # Warm start: zeros for fixed regs, μ = 0, lnσ = -1 (σ ≈ 0.37).
    θ0 = zeros(nparam)
    θ0[end] = -1.0
    converged = true
    θ̂  = θ0
    ll = -Inf
    try
        res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                             Optim.Options(g_tol = 1e-7, iterations = 4000);
                             autodiff = :forward)
        θ̂ = Optim.minimizer(res)
        ll = -Optim.minimum(res)
    catch e
        converged = false
        @warn "stata_mixlogit: optimisation failed in a flat region — " *
              "returning warm-start θ0 with NaN SEs." exception = e
    end

    # FD Hessian; wrap inv in try for the (common) singular case.
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
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h_[i]*h_[j])
            end
        end
        return H
    end
    V_cov = try
        Hs = LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂))
        try LinearAlgebra.inv(Hs)
        catch e
            e isa LinearAlgebra.SingularException ?
                LinearAlgebra.pinv(Matrix(Hs)) : rethrow(e)
        end
    catch
        fill(NaN, nparam, nparam)
    end
    se = sqrt.(max.(LinearAlgebra.diag(V_cov), 0.0))
    # Delta-method σ = exp(lnσ): dσ/dlnσ = σ. Adjust the σ row only.
    σ̂ = exp(θ̂[end])
    Jdelta = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    Jdelta[end, end] = σ̂
    V_struct = Jdelta * V_cov * Jdelta'
    se_struct = sqrt.(max.(LinearAlgebra.diag(V_struct), 0.0))

    β_fixed = Float64[θ̂[c] for c in 1:p_fixed]
    μ_r = θ̂[p_fixed + 1]

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
                       "Mixed (random-parameters) logit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Group variable: $(case_var)",
                       "Number of cases", commafmt(G))
        Printf.@printf("%-56s%-13s = %6d\n",
                       "Random regressor: $(random_reg)",
                       "Integration pts.", integration_pts)
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        println(ll_str)
        converged || println("WARNING: optimisation failed; values are warm-start θ0")
        println()
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        function _print(label, b, s)
            z = b / s
            pp = isnan(s) || s == 0 ? NaN :
                 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit * s; hi = b + crit * s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("Mean        |")
        for c in 1:p_fixed
            _print(string(fixed_regs[c]), θ̂[c], se_struct[c])
        end
        _print("$(random_reg) (mean)", μ_r, se_struct[p_fixed + 1])
        println("-"^13, "+", "-"^64)
        println("SD          |")
        _print("$(random_reg) (sd)", σ̂, se_struct[end])
        println("-"^78)
    end

    return (; β_fixed, μ = μ_r, σ = σ̂, θ̂, V = V_struct, se = se_struct, ll,
              n_obs = N, n_cases = G,
              fixed_regs = collect(fixed_regs), random_reg,
              integration_pts, converged, nparam)
end
