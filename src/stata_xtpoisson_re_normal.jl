# (deps provided by the StatEcon module)
import Optim

"""
    stata_xtpoisson_re_normal(df; depvar, regs, idvar, integration_pts=12,
                              level=0.95, quiet=false) -> NamedTuple

Stata-style `xtpoisson <depvar> <regs>, re normal` — random-effects
Poisson with **Gaussian** (instead of Gamma) random intercept:

    y_it | x_it, u_i  ~ Poisson(exp(x_it'β + u_i))
    u_i               ~ N(0, σ_u²)

Per-panel likelihood is integrated out by `integration_pts`-point
Gauss-Hermite quadrature (Stata default = 12). Joint MLE in
(β, lnσ_u²) by LBFGS with ForwardDiff autodiff. OIM SEs from finite-
difference Hessian.

Output mirrors Stata's `xtpoisson, re normal` block: header (Number of
obs / groups / Random effects u_i ~ Gaussian / Obs per group /
Integration method / Integration pts. / Wald chi² / Log likelihood /
Prob > chi²), coefficient table, `/lnsig2u` row, auxiliary `sigma_u`
row, and the trailing `LR test of sigma_u=0: chibar2(01)`.

Returns `(; β, V, se, coefnames, ln_σ2u, σ_u, se_ln_σ2u, se_σu,
σu_lo, σu_hi, ll, ll_pooled, LR, p_chibar, n, n_panels, T_min, T_max,
T_avg, Wald, Wald_p)`.
"""
function stata_xtpoisson_re_normal(df; depvar::Symbol,
                                   regs::AbstractVector{Symbol},
                                   idvar::Symbol,
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
                    ones(DataFrames.nrow(g))))
          for g in panels]
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    # Pooled Poisson warm start.
    X_all = reduce(vcat, [p.X for p in pd])
    y_all = reduce(vcat, [p.y for p in pd])
    m_pooled = GLM.glm(X_all, y_all, Distributions.Poisson(), GLM.LogLink())
    β0       = GLM.coef(m_pooled)
    ll_pooled = GLM.loglikelihood(m_pooled)

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
    n_internal = max(integration_pts, 32)
    gh_x, gh_w = _gauss_hermite(n_internal)

    # Per-panel log-likelihood: log ∫ Π_t Pois(y_it | exp(η_t + u)) · N(u; 0, σ²) du.
    # Sub u = σ·x·√2: (1/√π) Σ_q w_q · Π_t Pois(y_it | exp(η_t + σ·x_q·√2)).
    function negll(θ)
        β     = θ[1:k]
        ln_σ2 = θ[k + 1]
        σ_u   = exp(0.5 * ln_σ2)
        ll = zero(eltype(θ))
        for p in pd
            ηbase = p.X * β
            log_terms = Vector{eltype(θ)}(undef, length(gh_x))
            for q in eachindex(gh_x)
                z   = σ_u * gh_x[q] * sqrt(2.0)
                acc = zero(eltype(θ))
                for t in eachindex(p.y)
                    η = ηbase[t] + z
                    acc += p.y[t] * η - exp(η) -
                           _c18_loggamma(p.y[t] + 1)
                end
                log_terms[q] = log(gh_w[q]) + acc
            end
            mlt = maximum(log_terms)
            ll += mlt + log(sum(exp.(log_terms .- mlt))) - 0.5 * log(Base.pi)
        end
        return -ll
    end

    θ0 = vcat(β0, log(0.5))
    res = _c18_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-7, iterations = 2000))
    θ̂   = Optim.minimizer(res)
    β   = θ̂[1:k]
    ln_σ2u = θ̂[k + 1]
    σ_u    = exp(0.5 * ln_σ2u)
    ll     = -negll(θ̂)

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
    se_ln   = se_full[k + 1]
    V       = V_full[1:k, 1:k]
    se      = se_β

    z   = β ./ se
    pv  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    lns_lo = ln_σ2u - crit * se_ln
    lns_hi = ln_σ2u + crit * se_ln
    σu_lo  = exp(0.5 * lns_lo)
    σu_hi  = exp(0.5 * lns_hi)
    se_σu  = 0.5 * σ_u * se_ln

    slope_idx = 1:k - 1
    Wald   = β[slope_idx]' * LinearAlgebra.inv(V[slope_idx, slope_idx]) *
             β[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                   Wald)

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
                       "Random effects u_i ~ Gaussian",
                       "Obs per group:")
        Printf.@printf("%-53s%18s = %6d\n", "", "min", T_min)
        Printf.@printf("%-53s%18s = %6.1f\n", "", "avg", T_avg)
        Printf.@printf("%-53s%18s = %6d\n", "", "max", T_max)
        println()
        Printf.@printf("%-53s%-17s= %6d\n",
                       "Integration method: mvaghermite",
                       "Integration pts.", integration_pts)
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
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "/lnsig2u", g9(ln_σ2u; w=10), g9(se_ln; w=9), "",
                       g9(lns_lo; w=9), g9(lns_hi; w=10))
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "sigma_u", g9(σ_u; w=10), g9(se_σu; w=9), "",
                       g9(σu_lo; w=9), g9(σu_hi; w=10))
        println("-"^78)
        Printf.@printf("LR test of sigma_u=0: chibar2(01) = %.2f%13sProb >= chibar2 = %.3f\n",
                       LR, "", p_chibar)
    end

    return (; β, V, se, coefnames = cnames, ln_σ2u, σ_u,
              se_ln_σ2u = se_ln, se_σu, σu_lo, σu_hi,
              ll, ll_pooled, LR, p_chibar,
              n = n_obs, n_panels, T_min, T_max, T_avg,
              Wald, Wald_p, model = m_pooled)
end

