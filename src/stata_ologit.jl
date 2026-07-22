import Optim

"""
    stata_ologit(df, depvar, regs; level=0.95, quiet=false) -> NamedTuple

Stata `ologit <depvar> <regs>, nolog`. Ordered logit MLE on case-level
data. For y ∈ {1, …, J}:

    P(y_i = j) = Λ(τ_j − xᵢ'β) − Λ(τ_{j−1} − xᵢ'β)

with τ_0 = −∞, τ_J = +∞, and J−1 cut-points τ_1 < τ_2 < … < τ_{J−1}.

To keep the cut-points ordered during optimisation we parametrise
them as τ_1 = θ_{cut,1}, τ_j = τ_{j−1} + exp(θ_{cut,j}) for j ≥ 2 —
strictly increasing, smooth, autodiff-friendly. ML via LBFGS;
OIM vcov via FD Hessian; SEs for τ are delta-method-corrected back
to the original scale.

Returns `(; β, τ, V, se_β, se_τ, X, J, dfc, ll, ll_null, LR, LR_p,
pseudo_r2, n, nparam, depvar_name, regs)`.
"""
function stata_ologit(df::DataFrames.AbstractDataFrame, depvar::Symbol,
                     regs::AbstractVector{Symbol};
                     level::Float64 = 0.95,
                     quiet::Bool = false)
    cols = vcat([depvar], collect(regs))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c15_raw.(col))
        end
    end

    y_raw = [_c15_raw(v) for v in dfc[!, depvar]]
    cats  = sort(unique(y_raw)); J = length(cats)
    # Map y to 1..J in sorted order.
    y_idx = Int[findfirst(==(v), cats) for v in y_raw]
    X = hcat([Float64.(_c15_raw.(dfc[!, r])) for r in regs]...)
    N, k = size(X)
    nparam = k + (J - 1)

    # Stable logistic CDF: Λ(z) = 1/(1+exp(−z)); use log1p form.
    Λ(z) = 1 / (1 + exp(-z))

    # Unpack θ → (β, τ) with strictly increasing τ.
    function unpack(θ)
        β = θ[1:k]
        τ = zeros(eltype(θ), J - 1)
        τ[1] = θ[k + 1]
        for j in 2:(J - 1)
            τ[j] = τ[j - 1] + exp(θ[k + j])
        end
        return β, τ
    end

    function negll(θ)
        β, τ = unpack(θ)
        η = X * β
        ll = zero(eltype(θ))
        for i in 1:N
            j = y_idx[i]
            p = if j == 1
                Λ(τ[1] - η[i])
            elseif j == J
                1 - Λ(τ[J - 1] - η[i])
            else
                Λ(τ[j] - η[i]) - Λ(τ[j - 1] - η[i])
            end
            ll += log(max(p, eps(Float64)))
        end
        return -ll
    end

    # Warm start: β = 0, cut-points at the empirical quantile gaps.
    θ0 = zeros(nparam)
    cum = 0.0
    for j in 1:(J - 1)
        cum += sum(y_idx .<= j) / N
        τj_warm = log(cum / (1 - cum))
        if j == 1
            θ0[k + 1] = τj_warm
        else
            prev = θ0[k + j - 1]      # actually the *unpacked* prev
            prev = (j == 2) ? θ0[k + 1] : (θ0[k + 1] + sum(exp.(θ0[k+2:k+j-1])))
            θ0[k + j] = log(max(τj_warm - prev, 0.01))
        end
    end

    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 4000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    β̂, τ̂  = unpack(θ̂)
    ll = -Optim.minimum(res)

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

    # Delta-method Jacobian: τ_1 = θ_{k+1}, τ_j = θ_{k+1} + Σ_{m=2..j} exp(θ_{k+m})
    # ⇒ ∂τ_j / ∂θ_{k+1} = 1, ∂τ_j / ∂θ_{k+m} = exp(θ_{k+m}) for 2 ≤ m ≤ j, else 0.
    Jmat = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    for j in 1:(J - 1)
        for m in 1:j
            if m == 1
                Jmat[k + j, k + 1] = 1.0
            else
                Jmat[k + j, k + m] = exp(θ̂[k + m])
            end
        end
        # zero out the rest of row k+j past column k+j
        for m in (j + 1):(J - 1)
            Jmat[k + j, k + m] = 0.0
        end
    end
    V = Jmat * V_raw * Jmat'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β = se[1:k]; se_τ = se[k+1:end]

    # Null log-lik (intercept-only ordered logit) — fit just the cut-points.
    function negll_null(τθ)
        # τθ has length J-1, parametrised same way as full model's cut block.
        τ = zeros(eltype(τθ), J - 1)
        τ[1] = τθ[1]
        for j in 2:(J - 1); τ[j] = τ[j - 1] + exp(τθ[j]); end
        ll = zero(eltype(τθ))
        for i in 1:N
            j = y_idx[i]
            p = if j == 1; Λ(τ[1])
                elseif j == J; 1 - Λ(τ[J - 1])
                else; Λ(τ[j]) - Λ(τ[j - 1])
                end
            ll += log(max(p, eps(Float64)))
        end
        return -ll
    end
    τθ0 = θ0[k+1:end]
    res_null = _c15_optimize(negll_null, τθ0, Optim.LBFGS();
                              autodiff = :forward)
    # Re-evaluate at the (Float64) minimizer to guarantee a plain
    # Float64 LL — `Optim.minimum` sometimes returns a ForwardDiff.Dual
    # from a cached gradient pass, which then explodes inside
    # Distributions.cdf(Chisq, _) → gamma_inc (no Dual method).
    τθ_hat  = Optim.minimizer(res_null)
    ll_null = -negll_null(τθ_hat)
    LR      = 2 * (ll - ll_null)
    LR_p    = 1 - Distributions.cdf(Distributions.Chisq(k), LR)
    pseudo_r2 = 1 - ll / ll_null

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
                       "Ordered logistic regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6.2f\n", "", "LR chi2($k)", LR)
        Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", LR_p)
        ll_s = Printf.@sprintf("Log likelihood = %.4f", ll)
        r2_s = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad = max(0, 78 - length(ll_s) - length(r2_s))
        println(ll_s, " "^pad, r2_s); println()

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
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        for c in 1:k
            _print(string(regs[c]), β̂[c], se_β[c])
        end
        println("-"^13, "+", "-"^64)
        for j in 1:(J - 1)
            _print("/cut$j", τ̂[j], se_τ[j])
        end
        println("-"^78)
    end

    return (; β = β̂, τ = τ̂, V, se_β, se_τ, X, J, dfc,
              depvar_name = depvar, regs = collect(regs),
              ll, ll_null, LR, LR_p, pseudo_r2,
              n = N, nparam)
end

"""
    _c15_ologit_probs(β, τ, X, J) -> Matrix

Ordered-logit `predict, pr`: `P[i, j] = Λ(τ_j − xᵢ'β) − Λ(τ_{j-1} − xᵢ'β)`
with `τ_0 = −∞`, `τ_J = +∞`. `β`, `τ`, `X`, `J` come from a `stata_ologit`
fit; returns the `n × J` matrix of predicted category probabilities.
"""
function _c15_ologit_probs(β, τ, X, J)
    n = size(X, 1)
    P = zeros(n, J)
    Λ(z) = 1 / (1 + exp(-z))
    for i in 1:n
        η = LinearAlgebra.dot(view(X, i, :), β)
        P[i, 1] = Λ(τ[1] - η)
        for j in 2:(J - 1)
            P[i, j] = Λ(τ[j] - η) - Λ(τ[j - 1] - η)
        end
        P[i, J] = 1 - Λ(τ[J - 1] - η)
    end
    return P
end

"""
    stata_margins_dydx_ologit(res; outcome, at=:mean, level=0.95,
                              quiet=false)
        -> NamedTuple

Stata `margins, dydx(*) predict(outcome(<j>))` after `ologit`. Closed-
form partial derivative:

    ∂P(y = j) / ∂x_c = β_c · [λ(τ_{j-1} − x'β) − λ(τ_j − x'β)]

with τ_0 = −∞, τ_J = +∞, λ(z) = Λ(z)(1−Λ(z)). Two anchor policies:
`at = :mean` (Stata's `atmean`, MEM) or `at = :asobserved` (default,
AME). SE is delta-method via central-FD Jacobian of the closed-form
ME against `res.V` (which spans `(β, τ)` jointly).

Returns `(; rows, target_outcome, at, n)` where `rows` is a Vector
of NamedTuples `(reg, dydx, se, z, p, ci_lo, ci_hi)`.
"""
function stata_margins_dydx_ologit(res; outcome::Integer,
                                   at = :mean,
                                   level::Float64 = 0.95,
                                   quiet::Bool = false)
    k = size(res.X, 2); J = res.J
    1 <= outcome <= J ||
        error("outcome=$outcome out of range 1..$J")
    Xeval = at === :mean       ? reshape(vec(Statistics.mean(res.X; dims = 1)),
                                         1, k) :
            at === :asobserved ? res.X :
            reshape(convert(Vector{Float64}, at), 1, k)

    regs = collect(res.regs)
    Λ(z) = 1 / (1 + exp(-z))
    λ(z) = (l = Λ(z); l * (1 - l))

    function me_at(θ, c)
        β = view(θ, 1:k); τ = view(θ, (k + 1):(k + J - 1))
        Nrows = size(Xeval, 1); acc = zero(eltype(θ))
        for i in 1:Nrows
            xi = view(Xeval, i, :)
            η  = LinearAlgebra.dot(xi, β)
            d_lo = outcome == 1 ? zero(eltype(θ)) : λ(τ[outcome - 1] - η)
            d_hi = outcome == J ? zero(eltype(θ)) : λ(τ[outcome]      - η)
            acc += β[c] * (d_lo - d_hi)
        end
        return acc / Nrows
    end

    θ̂ = vcat(res.β, res.τ)
    crit = Distributions.quantile(Distributions.Normal(),
                                  1 - (1 - level) / 2)
    h_step = sqrt(eps(Float64))
    rows = NamedTuple[]
    for c in 1:k
        m_pt = me_at(θ̂, c)
        g = zeros(length(θ̂))
        for i in eachindex(θ̂)
            step = max(abs(θ̂[i]) * h_step, h_step)
            tp = copy(θ̂); tp[i] += step
            tm = copy(θ̂); tm[i] -= step
            g[i] = (me_at(tp, c) - me_at(tm, c)) / (2 * step)
        end
        se = sqrt(max(g' * res.V * g, 0.0))
        z  = m_pt / se
        p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        push!(rows, (; reg = regs[c], dydx = m_pt, se, z, p,
                      ci_lo = m_pt - crit * se, ci_hi = m_pt + crit * se))
    end

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
        N = size(res.X, 1)
        println()
        title = at === :asobserved ? "Average marginal effects" :
                                     "Conditional marginal effects"
        Printf.@printf("%-56s%-13s = %6s\n", title,
                       "Number of obs", commafmt(N))
        println("Model VCE: OIM")
        Printf.@printf("Expression: Pr(%s == %d), predict(outcome(%d))\n",
                       string(res.depvar_name), outcome, outcome)
        Printf.@printf("dy/dx wrt: %s\n", join(string.(regs), " "))
        if at === :mean
            xb = vec(Xeval[1, :])
            println("At: ",
                join((string(r) * " = " *
                      Printf.@sprintf("%.6g", xb[i]) * " (mean)"
                      for (i, r) in enumerate(regs)), "  "))
        end
        println()
        println("-"^78)
        println("             |            Delta-method")
        Printf.@printf("%12s |      dy/dx   Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)
        for r in rows
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           string(r.reg),
                           g9(r.dydx; w = 10), g9(r.se; w = 9),
                           Printf.@sprintf("%7.2f", r.z),
                           Printf.@sprintf("%.3f", r.p),
                           g9(r.ci_lo; w = 9), g9(r.ci_hi; w = 10))
        end
        println("-"^78)
    end
    return (; rows, target_outcome = outcome, at = Xeval, n = size(res.X, 1))
end
