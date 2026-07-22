# ============================================================================
# stata_xtivreg_fe.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

_c9_rawval(x) = hasproperty(x, :value) ? x.value : x

"""
    stata_xtivreg_fe(df, formula; y, xvars, endog, instruments, idvar, cluster, boot_β, level)

Stata-style `xtivreg ..., fe` — within-group 2SLS with a variance-component
footer (σ_u, σ_e, ρ, _cons) and an F-test that all individual effects are zero.
"""
function stata_xtivreg_fe(df, formula;
                    cluster = nothing,
                    y::Union{Symbol,Nothing} = nothing,
                    xvars::Union{Vector{Symbol},Nothing} = nothing,
                    endog::Union{Symbol,Nothing} = nothing,
                    instruments::Union{Vector{Symbol},Nothing} = nothing,
                    idvar::Union{Symbol,Nothing} = nothing,
                    boot_β = nothing,
                    level::Float64 = 0.95)
    m = cluster === nothing ?
        FixedEffectModels.reg(df, formula) :
        FixedEffectModels.reg(df, formula, FixedEffectModels.Vcov.cluster(cluster))

    # If footer arguments are not supplied, just return the model
    y === nothing && return m

    all_needed = unique(vcat([y, idvar], xvars,
                             instruments === nothing ? Symbol[] : instruments))
    d  = DataFrames.dropmissing(df, all_needed)
    β  = StatsBase.coef(m)
    cn = StatsBase.coefnames(m)
    ix = [findfirst(==(string(v)), cn) for v in xvars]
    gdf = DataFrames.groupby(d, idvar)
    n, N = DataFrames.nrow(d), length(gdf)
    k = length(cn)
    β_slope = β[ix]

    # ── Panel means ───────────────────────────────────────────────
    means = DataFrames.combine(gdf,
        [y => (x -> Statistics.mean(Float64.(x))) => :ȳ;
         [v => (x -> Statistics.mean(Float64.(x))) => Symbol(v, :_bar)
          for v in xvars]...])
    d2 = DataFrames.leftjoin(d, means, on = idvar)
    ỹ  = Float64.(d2[!, y]) .- d2.ȳ
    X̃  = hcat([Float64.(d2[!, v]) .- d2[!, Symbol(v, :_bar)] for v in xvars]...)
    e_w  = ỹ .- X̃ * β_slope
    σ2_e = sum(abs2, e_w) / (n - N - k)
    σ_e  = sqrt(σ2_e)

    # ── Fixed effects from IV β̂ → σ_u ────────────────────────────
    α̂ = [means.ȳ[i] -
          sum(β_slope[j] * means[!, Symbol(xvars[j], :_bar)][i]
              for j in eachindex(xvars))
          for i in 1:N]
    T_counts = [DataFrames.nrow(g) for g in gdf]
    T̄    = Statistics.mean(T_counts)
    T_min, T_max = minimum(T_counts), maximum(T_counts)
    σ2_u = max(Statistics.var(α̂) - σ2_e / T̄, 0.0)
    σ_u  = sqrt(σ2_u)
    ρ    = σ2_u / (σ2_u + σ2_e)
    β0   = Statistics.mean(α̂)

    # ── R² ───────────────────────────────────────────────────────
    ss_tot_w = sum(abs2, ỹ); ss_res_w = sum(abs2, e_w)
    r2_within = ss_tot_w > 0 ? 1 - ss_res_w/ss_tot_w : NaN
    Xβ_panel  = [sum(β_slope[j] * means[!, Symbol(xvars[j], :_bar)][i]
                     for j in eachindex(xvars)) for i in 1:N]
    r2_between = Statistics.cor(means.ȳ, Xβ_panel)^2
    y_all      = Float64.(d[!, y])
    Xβ_all     = hcat([Float64.(d[!, v]) for v in xvars]...) * β_slope
    r2_overall = Statistics.cor(y_all, Xβ_all)^2

    # ── corr(u_i, Xb) ────────────────────────────────────────────
    corr_ux = Statistics.cor(α̂, Xβ_panel)

    # ── SEs — bootstrap vs analytic ──────────────────────────────
    use_boot = boot_β !== nothing
    if use_boot
        B        = size(boot_β, 1)
        V_slope  = Statistics.cov(boot_β)
        se_slope = Statistics.std(boot_β, dims=1)[:]
        ȳ_grand  = Statistics.mean(y_all)
        X̄_grand  = [Statistics.mean(Float64.(d[!, v])) for v in xvars]
        β0_boot  = [ȳ_grand - sum(boot_β[b, j] * X̄_grand[j]
                                   for j in 1:length(xvars))
                    for b in 1:B]
        se_cons  = Statistics.std(β0_boot)
    else
        V_slope  = StatsBase.vcov(m)[ix, ix]
        se_slope = sqrt.(max.(LinearAlgebra.diag(V_slope), 0.0))
        se_cons  = NaN
    end
    z_slope = β_slope ./ se_slope
    p_slope = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_slope)))
    zcrit   = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo_s = β_slope .- zcrit .* se_slope
    ci_hi_s = β_slope .+ zcrit .* se_slope

    Wald   = β_slope' * LinearAlgebra.inv(V_slope) * β_slope
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(β_slope)), Wald)

    df1, df2 = N - 1, n - N - k
    F_val = Statistics.var(α̂) / σ2_e
    p_F   = 1 - Distributions.cdf(Distributions.FDist(df1, df2), F_val)

    # ── Stata-style formatters ──────────────────────────────────
    function _g(x, w; sig=7)
        (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return Printf.@sprintf("%*s", w, s)
    end
    function _c(v::Integer)
        s = string(abs(v)); neg = v < 0
        parts = String[]; i = length(s)
        while i >= 1
            push!(parts, s[max(1, i-2):i]); i -= 3
        end
        return (neg ? "-" : "") * join(reverse(parts), ",")
    end

    println()
    Printf.@printf("%-48s%-18s= %10s\n",
                   "Fixed-effects (within) IV regression",
                   "Number of obs", _c(n))
    Printf.@printf("%-48s%-18s= %10s\n",
                   "Group variable: $(idvar)", "Number of groups", _c(N))
    println()
    println("R-squared:                                      Obs per group:")
    within_str = (!isfinite(r2_within) || r2_within < 0) ?
                 "     ." : Printf.@sprintf("%6.4f", r2_within)
    Printf.@printf("     Within  = %7s                                  %-8s= %10d\n",
                   within_str, "min", T_min)
    Printf.@printf("     Between = %6.4f                                  %-8s= %10.1f\n",
                   r2_between, "avg", T̄)
    Printf.@printf("     Overall = %6.4f                                  %-8s= %10d\n",
                   r2_overall, "max", T_max)
    println()
    println()
    wlabel = Printf.@sprintf("Wald chi2(%d)", length(β_slope))
    Printf.@printf("%48s%-18s= %10.2f\n", " "^48, wlabel, Wald)
    left = Printf.@sprintf("corr(u_i, Xb) = %s", _g(corr_ux, 7; sig=4))
    Printf.@printf("%-48s%-18s= %10.4f\n", left, "Prob > chi2", Wald_p)
    println()

    if use_boot
        Printf.@printf("%s(Replications based on %d clusters in %s)\n",
                       " "^(78 - 40 - length(string(idvar)) - length(string(N))),
                       N, string(idvar))
    end
    println("-"^78)
    if use_boot
        Printf.@printf("%12s |   Observed   Bootstrap                         Normal-based\n", "")
        Printf.@printf("%12s | coefficient   std. err.      z    P>|z|     [%d%% conf. interval]\n",
                       string(y), round(Int, 100*level))
    else
        Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                       string(y), "Coefficient", "Std. err.", "z", "P>|z|",
                       round(Int, 100*level))
    end
    println("-"^13, "+", "-"^64)
    for j in eachindex(xvars)
        Printf.@printf("%12s | %s  %s  %6.2f  %5.3f  %s  %s\n",
                       string(xvars[j]),
                       _g(β_slope[j], 10), _g(se_slope[j], 9),
                       z_slope[j], p_slope[j],
                       _g(ci_lo_s[j], 11), _g(ci_hi_s[j], 11))
    end
    if use_boot
        z_c   = β0 / se_cons
        p_c   = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_c)))
        ci_lc = β0 - zcrit * se_cons
        ci_hc = β0 + zcrit * se_cons
        Printf.@printf("%12s | %s  %s  %6.2f  %5.3f  %s  %s\n",
                       "_cons", _g(β0, 10), _g(se_cons, 9),
                       z_c, p_c, _g(ci_lc, 11), _g(ci_hc, 11))
    else
        Printf.@printf("%12s | %s\n", "_cons", _g(β0, 10))
    end
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s\n", "sigma_u", _g(σ_u, 10))
    Printf.@printf("%12s | %s\n", "sigma_e", _g(σ_e, 10))
    Printf.@printf("%12s | %s   (fraction of variance due to u_i)\n",
                   "rho", _g(ρ, 10))
    println("-"^78)
    Printf.@printf(" F test that all u_i=0: F(%d,%d) = %8.2f            Prob > F    = %.4f\n",
                   df1, df2, F_val, p_F)
    println("-"^78)
    if endog !== nothing
        exog_names = setdiff(string.(xvars), [string(endog)])
        inst_names = instruments === nothing ? String[] : string.(instruments)
        println("Endogenous: ", string(endog))
        println("Exogenous:  ", join(vcat(exog_names, inst_names), " "))
    end

    return m
end
