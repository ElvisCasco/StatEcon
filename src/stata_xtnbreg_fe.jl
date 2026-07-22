# (deps provided by the StatEcon module)
import Optim

"""
    stata_xtnbreg_fe(df; depvar, regs, idvar, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xtnbreg <depvar> <regs>, fe i(<id>)` — Hausman-Hall-
Griliches (1984) **conditional fixed-effects** Negative Binomial.
The fixed effects are eliminated by conditioning on the panel total
`Σ_t y_it`. Closed-form conditional log-likelihood:

    log L_i = log Γ(Σ_t y_it + 1) + log Γ(Σ_t λ_it)
            − log Γ(Σ_t y_it + Σ_t λ_it)
            + Σ_t [log Γ(λ_it + y_it) − log Γ(λ_it) − log Γ(y_it+1)]

where `λ_it = exp(x_it'β)` (intercept retained — unlike conditional FE
logit where `_cons` is absorbed). Panels with `T_i < 2` are dropped
(uninformative under the conditioning). Joint MLE over β by Nelder-
Mead; OIM SEs from finite-difference Hessian.

Output mirrors Stata's `xtnbreg, fe` block: header (Number of obs /
groups / Obs per group / Wald chi² / Log likelihood / Prob > chi²)
and the coefficient table.

Returns `(; β, V, se, coefnames, ll, n, n_panels, T_min, T_max,
T_avg, Wald, Wald_p)`.
"""
function stata_xtnbreg_fe(df; depvar::Symbol,
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

    panels_all = DataFrames.groupby(d, idvar)
    pd_all = [(y = Float64.(g[!, depvar]),
               X = hcat([Float64.(g[!, v]) for v in regs]...,
                        ones(DataFrames.nrow(g))))
              for g in panels_all]

    # Drop panels that are uninformative under the conditioning:
    #   T_i < 2 (only one obs)  OR  Σ_t y_it == 0 (all zeros).
    pd = filter(p -> length(p.y) >= 2 && sum(p.y) > 0, pd_all)
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    # Pooled NB warm start (same depvar / regs).
    nb_pooled = stata_nbreg(d,
                StatsModels.term(depvar) ~
                    sum(StatsModels.term.(regs));
                quiet = true)
    β0 = vcat([nb_pooled.β_glm[findfirst(==(string(v)), nb_pooled.coefnames_glm)]
               for v in regs]...,
              nb_pooled.β_glm[findfirst(==("(Intercept)"),
                                        nb_pooled.coefnames_glm)])

    function negll(β)
        ll = zero(eltype(β))
        for p in pd
            λ      = exp.(p.X * β)
            sum_λ  = sum(λ)
            sum_y  = sum(p.y)
            ll_i = _c18_loggamma(sum_y + 1) +
                   _c18_loggamma(sum_λ) -
                   _c18_loggamma(sum_y + sum_λ)
            for t in eachindex(p.y)
                ll_i += _c18_loggamma(λ[t] + p.y[t]) -
                        _c18_loggamma(λ[t]) -
                        _c18_loggamma(p.y[t] + 1)
            end
            ll += ll_i
        end
        return -ll
    end

    res = _c18_optimize(negll, β0, Optim.NelderMead(),
                         Optim.Options(g_tol = 1e-9,
                                       x_abstol = 1e-9,
                                       f_reltol = 1e-12,
                                       iterations = 5000))
    β  = Optim.minimizer(res)
    ll = -negll(β)

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
    H = _fd_hessian(negll, β)
    V = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    z   = β ./ se
    pv  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

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
                       "Conditional FE negative binomial regression",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        println()
        Printf.@printf("%-53s%s\n", "", "Obs per group:")
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
        println("-"^78)
    end

    return (; β, V, se, coefnames = cnames, ll,
              n = n_obs, n_panels, T_min, T_max, T_avg,
              Wald, Wald_p)
end

