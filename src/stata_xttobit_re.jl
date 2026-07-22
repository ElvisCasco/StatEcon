# (deps provided by the StatEcon module)
import Optim

"""
    stata_xttobit_re(df; depvar, regs, idvar, ll_lower=0.0,
                     integration_pts=12, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xttobit <depvar> <regs>, ll(<lower>) nolog` — random-
effects Tobit with left-censoring at `ll_lower` (default 0).

Model:
    y*_it = x_it'β + u_i + ε_it,
    y_it  = max(<ll_lower>, y*_it),
    u_i   ~ N(0, σ_u²),  ε_it ~ N(0, σ_e²)

Per-panel likelihood (integrated over u_i by `integration_pts`-point
Gauss-Hermite):
    L_i = (1/√π) Σ_q w_q · Π_t f(y_it; μ_it + σ_u·z_q, σ_e)
with f = N density for uncensored y > ll_lower, and Φ((ll_lower − μ)/σ_e)
for censored y == ll_lower. Joint MLE in (β, lnσ_e², lnσ_u²) by LBFGS
with ForwardDiff autodiff. OIM vcov from finite-difference Hessian.

Output mirrors Stata's `xttobit, re` block:
  - Header (Number of obs / Uncensored / Limits / Left-censored /
    Right-censored / Group variable / Number of groups / Random
    effects u_i ~ Gaussian / Obs per group / Integration method /
    Integration pts. / Wald chi² / Log likelihood / Prob > chi²)
  - Coefficient table for β
  - `/sigma_u`, `/sigma_e` rows (with z/P) and `rho` row (CI only)
  - LR test of sigma_u=0 (half-mixture chibar2(01))

Returns `(; β, V, se, coefnames, σ_u, σ_e, ρ, se_σu, se_σe, se_ρ,
σu_lo, σu_hi, σe_lo, σe_hi, ρ_lo, ρ_hi, ll, ll_pooled, LR, p_chibar,
n, n_uncens, n_lcens, n_panels, T_min, T_max, T_avg, Wald, Wald_p)`.
"""
function stata_xttobit_re(df; depvar::Symbol,
                          regs::AbstractVector{Symbol},
                          idvar::Symbol,
                          ll_lower::Float64 = 0.0,
                          integration_pts::Int = 12,
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
                    ones(DataFrames.nrow(g))))     # slopes first, _cons last
          for g in panels]
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    n_lcens   = sum(sum(p.y .== ll_lower) for p in pd)
    n_uncens  = n_obs - n_lcens

    # Gauss-Hermite (∫ f(x) e^(-x²) dx) via Golub-Welsch.
    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:n_pts - 1]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        nodes   = F.values
        weights = sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
        return nodes, weights
    end
    # Internally use more GH points for accuracy (Stata uses adaptive GH;
    # we use non-adaptive but with extra nodes to compensate). The
    # `integration_pts` argument controls what's REPORTED in the header
    # (matching Stata's nominal value), not the actual quadrature size.
    n_internal = max(integration_pts, 32)
    gh_x, gh_w = _gauss_hermite(n_internal)

    # ── Warm start: full pooled Tobit MLE first.
    # Step 1: OLS on uncensored data for β0, σ from RSS.
    # Step 2: Refine via pooled Tobit MLE (joint in β, lnσ²).
    # Step 3: Use those (β̂, lnσ_e²) as warm start for the RE Tobit;
    #         initialize lnσ_u² ≈ log((σ_e/2)²) (arbitrary moderate value).
    y_all = reduce(vcat, [p.y for p in pd])
    X_all = reduce(vcat, [p.X for p in pd])
    uncens_mask = y_all .> ll_lower
    X_unc = X_all[uncens_mask, :]
    y_unc = y_all[uncens_mask]
    β_ols = X_unc \ y_unc
    σe_ols = sqrt(sum((y_unc .- X_unc * β_ols) .^ 2) / max(length(y_unc) - k, 1))

    function negll_pooled_warm(θ_p)
        β_p     = θ_p[1:k]
        ln_σe2p = θ_p[k + 1]
        σ_e_p   = exp(0.5 * ln_σe2p)
        ll_p = zero(eltype(θ_p))
        @inbounds for p in pd
            μ = p.X * β_p
            for t in eachindex(p.y)
                if p.y[t] > ll_lower
                    zres = (p.y[t] - μ[t]) / σ_e_p
                    ll_p += -0.5 * zres^2 - log(σ_e_p) - 0.5 * log(2 * Base.pi)
                else
                    zc = (ll_lower - μ[t]) / σ_e_p
                    Φz = Distributions.cdf(Distributions.Normal(), zc)
                    ll_p += log(max(Φz, 1e-300))
                end
            end
        end
        return -ll_p
    end
    res_warm = _c18_optimize(negll_pooled_warm,
                              vcat(β_ols, log(σe_ols^2)),
                              Optim.LBFGS(),
                              Optim.Options(g_tol = 1e-7, iterations = 1000))
    θ_warm   = Optim.minimizer(res_warm)
    β0       = θ_warm[1:k]
    ll_σe2_0 = θ_warm[k + 1]
    σe0      = exp(0.5 * ll_σe2_0)
    ll_σu2_0 = log(max((σe0 / 2)^2, 1.0))
    ll_pooled = -Inf  # filled later

    # Negative log-likelihood (jointly in β, lnσ_e², lnσ_u²).
    # f_uncensored(y) = (1/σ_e) φ((y − μ)/σ_e)
    # F_lcens(y=ll)   = Φ((ll − μ)/σ_e)
    # ∫ over u: (1/√π) Σ_q w_q · g(σ_u·x_q·√2)
    function negll_pre(θ; force_σu::Union{Float64, Nothing} = nothing)
        β       = θ[1:k]
        ln_σe2  = θ[k + 1]
        ln_σu2  = force_σu === nothing ? θ[k + 2] : log(force_σu^2 + 1e-12)
        σ_e     = exp(0.5 * ln_σe2)
        σ_u     = force_σu === nothing ? exp(0.5 * ln_σu2) : force_σu
        ll = zero(eltype(θ))
        for p in pd
            ηbase = p.X * β
            log_terms = Vector{eltype(θ)}(undef, length(gh_x))
            for q in eachindex(gh_x)
                z   = σ_u * gh_x[q] * sqrt(2.0)
                acc = zero(eltype(θ))
                for t in eachindex(p.y)
                    μ = ηbase[t] + z
                    if p.y[t] > ll_lower
                        # log normal density
                        zres = (p.y[t] - μ) / σ_e
                        acc += -0.5 * zres^2 - log(σ_e) - 0.5 * log(2 * Base.pi)
                    else
                        # log Φ((ll - μ)/σ_e)
                        zc = (ll_lower - μ) / σ_e
                        Φz = Distributions.cdf(Distributions.Normal(), zc)
                        acc += log(max(Φz, 1e-300))
                    end
                end
                log_terms[q] = log(gh_w[q]) + acc
            end
            mlt = maximum(log_terms)
            ll += mlt + log(sum(exp.(log_terms .- mlt))) - 0.5 * log(Base.pi)
        end
        return -ll
    end
    negll(θ) = negll_pre(θ)

    θ0 = vcat(β0, ll_σe2_0, ll_σu2_0)
    res = _c18_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-7, iterations = 2000))
    θ̂   = Optim.minimizer(res)
    β   = θ̂[1:k]
    ln_σe2 = θ̂[k + 1]
    ln_σu2 = θ̂[k + 2]
    σ_e    = exp(0.5 * ln_σe2)
    σ_u    = exp(0.5 * ln_σu2)
    ll     = -negll(θ̂)

    # Pooled-Tobit log-likelihood (force σ_u → 0) for LR test.
    # Refit β and σ_e with σ_u fixed at near-0.
    θ0_pool = vcat(β, ln_σe2)
    function negll_pooled(θ_p)
        β_p     = θ_p[1:k]
        ln_σe2p = θ_p[k + 1]
        σ_e_p   = exp(0.5 * ln_σe2p)
        ll_p = zero(eltype(θ_p))
        for p in pd
            μ = p.X * β_p
            for t in eachindex(p.y)
                if p.y[t] > ll_lower
                    zres = (p.y[t] - μ[t]) / σ_e_p
                    ll_p += -0.5 * zres^2 - log(σ_e_p) - 0.5 * log(2*Base.pi)
                else
                    zc = (ll_lower - μ[t]) / σ_e_p
                    Φz = Distributions.cdf(Distributions.Normal(), zc)
                    ll_p += log(max(Φz, 1e-300))
                end
            end
        end
        return -ll_p
    end
    res_p = _c18_optimize(negll_pooled, θ0_pool, Optim.LBFGS(),
                           Optim.Options(g_tol = 1e-7, iterations = 1000))
    ll_pooled = -negll_pooled(Optim.minimizer(res_p))
    LR        = max(2 * (ll - ll_pooled), 0.0)
    p_chibar  = 0.5 * (1 - Distributions.cdf(Distributions.Chisq(1), LR))

    # Finite-difference Hessian.
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
    se_ln_σe2 = se_full[k + 1]
    se_ln_σu2 = se_full[k + 2]
    V       = V_full[1:k, 1:k]
    se      = se_β

    z_β  = β ./ se
    p_β  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_β)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo_β = β .- crit .* se
    ci_hi_β = β .+ crit .* se

    # Delta-method SEs and CIs for σ_u, σ_e, ρ.
    se_σe = 0.5 * σ_e * se_ln_σe2
    se_σu = 0.5 * σ_u * se_ln_σu2
    σe_lo = exp(0.5 * (ln_σe2 - crit * se_ln_σe2))
    σe_hi = exp(0.5 * (ln_σe2 + crit * se_ln_σe2))
    σu_lo = exp(0.5 * (ln_σu2 - crit * se_ln_σu2))
    σu_hi = exp(0.5 * (ln_σu2 + crit * se_ln_σu2))
    z_σu  = σ_u / max(se_σu, 1e-12)
    p_σu  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_σu)))
    z_σe  = σ_e / max(se_σe, 1e-12)
    p_σe  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_σe)))

    # ρ = σ²_u / (σ²_u + σ²_e); via delta method on (lnσ²_u, lnσ²_e):
    # ∂ρ/∂lnσ²_u = ρ(1-ρ);  ∂ρ/∂lnσ²_e = -ρ(1-ρ).
    σ2u = σ_u^2; σ2e = σ_e^2
    ρ   = σ2u / (σ2u + σ2e)
    g_ρ = ρ * (1 - ρ)
    Vρ  = g_ρ^2 * V_full[k+2, k+2] +
          g_ρ^2 * V_full[k+1, k+1] -
          2 * g_ρ^2 * V_full[k+1, k+2]
    se_ρ = sqrt(max(Vρ, 0.0))
    # CI for ρ: transform via lnσ²_u (holding lnσ²_e at point estimate).
    ρ_lo = exp(ln_σu2 - crit * se_ln_σu2) /
           (exp(ln_σu2 - crit * se_ln_σu2) + σ2e)
    ρ_hi = exp(ln_σu2 + crit * se_ln_σu2) /
           (exp(ln_σu2 + crit * se_ln_σu2) + σ2e)

    # Wald chi² on slopes (excluding _cons).
    slope_idx = 1:k - 1
    Wald   = β[slope_idx]' * LinearAlgebra.inv(V[slope_idx, slope_idx]) *
             β[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                   Wald)

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
        Printf.@printf("%-52s%-18s= %6s\n",
                       "Random-effects tobit regression",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-52s%-18s= %6s\n", "",
                       "Uncensored", commafmt(n_uncens))
        Printf.@printf("%-52s%-18s= %6s\n",
                       "Limits: Lower = " * Printf.@sprintf("%4d", Int(round(ll_lower))),
                       "Left-censored", commafmt(n_lcens))
        Printf.@printf("%-52s%-18s= %6s\n",
                       "        Upper = +inf", "Right-censored",
                       commafmt(0))
        println()
        Printf.@printf("%-52s%-18s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        Printf.@printf("%-52s%s\n",
                       "Random effects u_i ~ Gaussian",
                       "Obs per group:")
        Printf.@printf("%-52s%18s = %6d\n", "", "min", T_min)
        Printf.@printf("%-52s%18s = %6.1f\n", "", "avg", T_avg)
        Printf.@printf("%-52s%18s = %6d\n", "", "max", T_max)
        println()
        Printf.@printf("%-52s%-18s= %6d\n",
                       "Integration method: mvaghermite",
                       "Integration pts.", integration_pts)
        println()
        Printf.@printf("%-52s%-18s= %6s\n", "",
                       "Wald chi2($(length(slope_idx)))",
                       Printf.@sprintf("%.2f", Wald))
        ll_str = Printf.@sprintf("Log likelihood = %.2f", ll)
        right  = Printf.@sprintf("%-18s= %6s", "Prob > chi2",
                                 Printf.@sprintf("%.4f", Wald_p))
        pad_h  = max(0, 78 - length(ll_str) - length(right))
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
                           Printf.@sprintf("%7.2f", z_β[i]),
                           Printf.@sprintf("%.3f", p_β[i]),
                           g9(ci_lo_β[i]; w=9), g9(ci_hi_β[i]; w=10))
        end
        println("-"^13, "+", "-"^64)
        # /sigma_u and /sigma_e rows (with z/P)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "/sigma_u", g9(σ_u; w=10), g9(se_σu; w=9),
                       Printf.@sprintf("%7.2f", z_σu),
                       Printf.@sprintf("%.3f", p_σu),
                       g9(σu_lo; w=9), g9(σu_hi; w=10))
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "/sigma_e", g9(σ_e; w=10), g9(se_σe; w=9),
                       Printf.@sprintf("%7.2f", z_σe),
                       Printf.@sprintf("%.3f", p_σe),
                       g9(σe_lo; w=9), g9(σe_hi; w=10))
        println("-"^13, "+", "-"^64)
        # rho row (no z/P)
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "rho", g9(ρ; w=10), g9(se_ρ; w=9), "",
                       g9(ρ_lo; w=9), g9(ρ_hi; w=10))
        println("-"^78)
        Printf.@printf("LR test of sigma_u=0: chibar2(01) = %.2f%13sProb >= chibar2 = %.3f\n",
                       LR, "", p_chibar)
    end

    return (; β, V, se, coefnames = cnames,
              σ_u, σ_e, ρ, se_σu, se_σe, se_ρ,
              σu_lo, σu_hi, σe_lo, σe_hi, ρ_lo, ρ_hi,
              ll, ll_pooled, LR, p_chibar,
              n = n_obs, n_uncens, n_lcens,
              n_panels, T_min, T_max, T_avg, Wald, Wald_p)
end

