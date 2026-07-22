# (deps provided by the StatEcon module)
import Optim

"""
    stata_xtpoisson_re(df; depvar, regs, idvar, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xtpoisson <depvar> <regs>, re` — Hausman-Hall-Griliches
(1984) **Gamma random-effects** Poisson:

    y_it | x_it, ν_i  ~ Poisson(μ_it · ν_i),    μ_it = exp(x_it'β)
    ν_i               ~ Gamma(1/α, 1/α)         (mean 1, variance α)

The mixture has a closed-form panel likelihood (no Gauss-Hermite needed):

    ℓ_i = Σ_t [y_it · ln μ_it − ln Γ(y_it+1)]
          + ln Γ(Σy_it + 1/α) − ln Γ(1/α)
          + (1/α) · ln(1/α) − (Σy_it + 1/α) · ln(Σμ_it + 1/α)

Joint MLE in (β, lnα) by LBFGS with ForwardDiff autodiff. OIM SEs from
finite-difference Hessian. The LR test of `α=0` is `chibar2(01)`
(half-mixture, since α=0 is on the boundary).

Output mirrors Stata's `xtpoisson, re`: header (Number of obs / groups
/ Random effects u_i ~ Gamma / Obs per group / Wald chi² / Log
likelihood / Prob > chi²), coefficient table for β, `/lnalpha` row,
auxiliary `alpha` row with delta-method CIs, and the trailing
`LR test of alpha=0: chibar2(01)` line.

Returns `(; β, V, se, coefnames, lnα, α, se_lnα, se_α, αlo, αhi,
ll, ll_pooled, LR, p_chibar, n, n_panels, T_min, T_max, T_avg,
Wald, Wald_p)`.
"""
function stata_xtpoisson_re(df; depvar::Symbol,
                            regs::AbstractVector{Symbol},
                            idvar::Symbol,
                            level::Float64 = 0.95,
                            quiet::Bool = false)
    needed = unique(vcat(depvar, idvar, regs))
    d = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = d[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            d[!, c] = Float64.(col)
        end
    end
    keep = trues(DataFrames.nrow(d))
    for c in needed
        col = d[!, c]
        eltype(col) <: Real || continue
        for i in eachindex(col); keep[i] &= isfinite(col[i]); end
    end
    d = d[keep, :]
    d = DataFrames.sort(d, [idvar])

    panels = DataFrames.groupby(d, idvar)
    pd = [(y = Float64.(g[!, depvar]),
           X = hcat([Float64.(g[!, v]) for v in regs]...,
                    ones(DataFrames.nrow(g))))
          for g in panels]
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    # Pooled-Poisson warm start.
    X_all = reduce(vcat, [p.X for p in pd])
    y_all = reduce(vcat, [p.y for p in pd])
    m_pooled = GLM.glm(X_all, y_all, Distributions.Poisson(), GLM.LogLink())
    β0       = GLM.coef(m_pooled)
    ll_pooled = GLM.loglikelihood(m_pooled)

    # Negative log-likelihood: jointly in (β, lnα).
    function negll(θ)
        β   = θ[1:k]
        lnα = θ[k + 1]
        α   = exp(lnα)
        δ   = 1 / α
        ll = zero(eltype(θ))
        for p in pd
            μ      = exp.(p.X * β)
            sum_y  = sum(p.y)
            sum_μ  = sum(μ)
            ll_i = zero(eltype(θ))
            for t in eachindex(p.y)
                ll_i += p.y[t] * log(μ[t]) -
                        _c18_loggamma(p.y[t] + 1)
            end
            ll_i += _c18_loggamma(sum_y + δ) -
                    _c18_loggamma(δ) +
                    δ * log(δ) - (sum_y + δ) * log(sum_μ + δ)
            ll += ll_i
        end
        return -ll
    end

    θ0 = vcat(β0, log(1.0))
    res = _c18_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 2000))
    θ̂  = Optim.minimizer(res)
    β  = θ̂[1:k]
    lnα = θ̂[k + 1]
    α   = exp(lnα)
    ll  = -negll(θ̂)

    # Finite-difference Hessian for SE.
    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x)
            xpi[i] += h_[i]; xmi[i] -= h_[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h_[i]^2
            for j in (i+1):nθ
                xpp = copy(x); xpp[i] += h_[i]; xpp[j] += h_[j]
                xpm = copy(x); xpm[i] += h_[i]; xpm[j] -= h_[j]
                xmp = copy(x); xmp[i] -= h_[i]; xmp[j] += h_[j]
                xmm = copy(x); xmm[i] -= h_[i]; xmm[j] -= h_[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h_[i] * h_[j])
            end
        end
        return H
    end
    H = _fd_hessian(negll, θ̂)
    V_full = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))
    se_full = sqrt.(max.(LinearAlgebra.diag(V_full), 0.0))
    se_β    = se_full[1:k]
    se_lnα  = se_full[k + 1]
    V       = V_full[1:k, 1:k]
    se      = se_β

    z   = β ./ se
    pv  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    # /lnalpha CI directly; alpha via delta method.
    lnα_lo = lnα - crit * se_lnα
    lnα_hi = lnα + crit * se_lnα
    αlo    = exp(lnα_lo)
    αhi    = exp(lnα_hi)
    se_α   = α * se_lnα

    # Wald chi² on slopes (excludes _cons).
    slope_idx = 1:k - 1
    Wald   = β[slope_idx]' * LinearAlgebra.inv(V[slope_idx, slope_idx]) *
             β[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                   Wald)

    # LR test of α=0 (RE → pooled Poisson).
    LR       = max(2 * (ll - ll_pooled), 0.0)
    p_chibar = 0.5 * (1 - Distributions.cdf(Distributions.Chisq(1), LR))

    T_per = [length(p.y) for p in pd]
    T_min = minimum(T_per); T_max = maximum(T_per)
    T_avg = Statistics.mean(T_per)

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w)
    end
    commafmt(num) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    if !quiet
        println()
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Random-effects Poisson regression",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        println()
        Printf.@printf("%-53s%s\n",
                       "Random effects u_i ~ Gamma",
                       "Obs per group:")
        Printf.@printf("%-53s%18s = %6d\n", "", "min", T_min)
        Printf.@printf("%-53s%18s = %6.1f\n", "", "avg", T_avg)
        Printf.@printf("%-53s%18s = %6d\n", "", "max", T_max)
        println()
        Printf.@printf("%-53s%-17s= %6s\n", "",
                       "Wald chi2($(length(slope_idx)))",
                       Printf.@sprintf("%.2f", Wald))
        ll_str  = Printf.@sprintf("Log likelihood = %.3f", ll)
        right   = Printf.@sprintf("%-17s= %6s", "Prob > chi2",
                                  Printf.@sprintf("%.4f", Wald_p))
        pad_h   = max(0, 78 - length(ll_str) - length(right))
        println(ll_str, " "^pad_h, right)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100*level)
        println("-"^13, "+", "-"^64)
        for i in vcat(collect(slope_idx), [k])
            label = i == k ? "_cons" : cnames[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(β[i]; w=10), g9(se[i]; w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", pv[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^13, "+", "-"^64)
        # /lnalpha row
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "/lnalpha", g9(lnα; w=10), g9(se_lnα; w=9), "",
                       g9(lnα_lo; w=9), g9(lnα_hi; w=10))
        println("-"^13, "+", "-"^64)
        # alpha row (no z/P)
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "alpha", g9(α; w=10), g9(se_α; w=9), "",
                       g9(αlo; w=9), g9(αhi; w=10))
        println("-"^78)
        chibar_str = LR >= 1e4 ? Printf.@sprintf("%.1e", LR) :
                                  Printf.@sprintf("%.2f", LR)
        Printf.@printf("LR test of alpha=0: chibar2(01) = %-15s Prob >= chibar2 = %.3f\n",
                       chibar_str, p_chibar)
    end

    return (; β, V, se, coefnames = cnames, lnα, α, se_lnα, se_α,
              αlo, αhi, ll, ll_pooled, LR, p_chibar,
              n = n_obs, n_panels, T_min, T_max, T_avg,
              Wald, Wald_p, model = m_pooled)
end

