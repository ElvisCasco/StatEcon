# (deps provided by the StatEcon module)
import Optim

"""
    stata_xtnbreg_re(df; depvar, regs, idvar, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xtnbreg <depvar> <regs>, re i(<id>)` — Hausman-Hall-
Griliches (1984) **Beta random-effects** Negative Binomial:

    y_it | x_it, δ_i  ~ NB(λ_it, δ_i),    λ_it = exp(x_it'β)
    ν_i = δ_i/(1+δ_i) ~ Beta(r, s)        (random effects)

Closed-form panel log-likelihood:

    log L_i = log Γ(r+s) − log Γ(r) − log Γ(s)
            + log Γ(r + Σ_t λ_it) + log Γ(s + Σ_t y_it)
            − log Γ(r + s + Σ_t λ_it + Σ_t y_it)
            + Σ_t [log Γ(λ_it + y_it) − log Γ(λ_it) − log Γ(y_it + 1)]

Joint MLE in (β, ln r, ln s) by LBFGS with ForwardDiff autodiff. OIM
SEs from finite-difference Hessian. The LR test compares against the
pooled NB log-likelihood (obtained by re-fitting `stata_nbreg` on the
stacked data).

Output mirrors Stata's `xtnbreg, re`: header with `Random effects u_i ~
Beta`, coefficient table, `/ln_r` + `/ln_s` rows, auxiliary `r` + `s`
rows with delta-method CIs, and the trailing `LR test vs. pooled:
chibar2(01)` half-mixture line.

Returns `(; β, V, se, coefnames, ln_r, ln_s, r, s, se_ln_r, se_ln_s,
se_r, se_s, ll, ll_pooled, LR, p_chibar, n, n_panels, T_min, T_max,
T_avg, Wald, Wald_p)`.
"""
function stata_xtnbreg_re(df; depvar::Symbol,
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

    # Pooled NB warm start (β + a tiny α to avoid Poisson saturation).
    nb_pooled = stata_nbreg(d,
                StatsModels.term(depvar) ~
                    sum(StatsModels.term.(regs));
                quiet = true)
    # Reorder β to GLM order ([_cons, regs...]) — stata_nbreg returns
    # Stata-display order (slopes first, _cons last). Our local
    # X = [regs..., ones] matches Stata-display order for slopes + cons.
    β0 = vcat([nb_pooled.β_glm[findfirst(==(string(v)), nb_pooled.coefnames_glm)]
               for v in regs]...,
              nb_pooled.β_glm[findfirst(==("(Intercept)"),
                                        nb_pooled.coefnames_glm)])
    ll_pooled = nb_pooled.ll

    # Negative log-likelihood (jointly in β, ln_r, ln_s).
    # log L_i = log Γ(r+s) − log Γ(r) − log Γ(s)
    #        + log Γ(r + Σλ) + log Γ(s + Σy)
    #        − log Γ(r + s + Σλ + Σy)
    #        + Σ_t [log Γ(λ_it + y_it) − log Γ(λ_it) − log Γ(y_it + 1)]
    function negll(θ)
        β    = θ[1:k]
        ln_r = θ[k + 1]
        ln_s = θ[k + 2]
        r    = exp(ln_r)
        s    = exp(ln_s)
        ll = zero(eltype(θ))
        for p in pd
            λ      = exp.(p.X * β)
            sum_λ  = sum(λ)
            sum_y  = sum(p.y)
            ll_i = _c18_loggamma(r + s) -
                   _c18_loggamma(r) -
                   _c18_loggamma(s) +
                   _c18_loggamma(r + sum_λ) +
                   _c18_loggamma(s + sum_y) -
                   _c18_loggamma(r + s + sum_λ + sum_y)
            for t in eachindex(p.y)
                ll_i += _c18_loggamma(λ[t] + p.y[t]) -
                        _c18_loggamma(λ[t]) -
                        _c18_loggamma(p.y[t] + 1)
            end
            ll += ll_i
        end
        return -ll
    end

    θ0 = vcat(β0, log(2.0), log(2.0))
    res = _c18_optimize(negll, θ0, Optim.NelderMead(),
                         Optim.Options(g_tol = 1e-9,
                                       x_abstol = 1e-9,
                                       f_reltol = 1e-12,
                                       iterations = 5000))
    θ̂  = Optim.minimizer(res)
    β  = θ̂[1:k]
    ln_r = θ̂[k + 1]
    ln_s = θ̂[k + 2]
    r    = exp(ln_r)
    s    = exp(ln_s)
    ll   = -negll(θ̂)

    # OIM vcov via FD Hessian (loggamma is FD-friendly via ForwardDiff
    # but using FD here keeps consistency with `stata_nbreg`).
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
    se_ln_r = se_full[k + 1]
    se_ln_s = se_full[k + 2]
    V       = V_full[1:k, 1:k]
    se      = se_β

    z   = β ./ se
    pv  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    # /ln_r and /ln_s CIs directly; r, s via delta method on exp().
    ln_r_lo = ln_r - crit * se_ln_r;  ln_r_hi = ln_r + crit * se_ln_r
    ln_s_lo = ln_s - crit * se_ln_s;  ln_s_hi = ln_s + crit * se_ln_s
    r_lo, r_hi = exp(ln_r_lo), exp(ln_r_hi)
    s_lo, s_hi = exp(ln_s_lo), exp(ln_s_hi)
    se_r = r * se_ln_r
    se_s = s * se_ln_s

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
        s_str = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s_str) > cap && sig_use > 1
            sig_use -= 1
            s_str = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s_str = replace(s_str, r"^(-?)0\." => s"\1."))
        lpad(s_str, w)
    end
    commafmt(num) = begin
        s_str = string(abs(num)); parts = String[]; i = length(s_str)
        while i >= 1; push!(parts, s_str[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    if !quiet
        println()
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Random-effects negative binomial regression",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        println()
        Printf.@printf("%-53s%s\n",
                       "Random effects u_i ~ Beta",
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
        # /ln_r and /ln_s rows
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "/ln_r", g9(ln_r; w=10), g9(se_ln_r; w=9), "",
                       g9(ln_r_lo; w=9), g9(ln_r_hi; w=10))
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "/ln_s", g9(ln_s; w=10), g9(se_ln_s; w=9), "",
                       g9(ln_s_lo; w=9), g9(ln_s_hi; w=10))
        println("-"^13, "+", "-"^64)
        # r and s rows
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "r", g9(r; w=10), g9(se_r; w=9), "",
                       g9(r_lo; w=9), g9(r_hi; w=10))
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "s", g9(s; w=10), g9(se_s; w=9), "",
                       g9(s_lo; w=9), g9(s_hi; w=10))
        println("-"^78)
        chibar_str = LR >= 1e4 ? Printf.@sprintf("%.1e", LR) :
                                  Printf.@sprintf("%.2f", LR)
        Printf.@printf("LR test vs. pooled: chibar2(01) = %-15s Prob >= chibar2 = %.3f\n",
                       chibar_str, p_chibar)
    end

    return (; β, V, se, coefnames = cnames,
              ln_r, ln_s, r, s, se_ln_r, se_ln_s, se_r, se_s,
              r_lo, r_hi, s_lo, s_hi,
              ll, ll_pooled, LR, p_chibar,
              n = n_obs, n_panels, T_min, T_max, T_avg,
              Wald, Wald_p)
end

