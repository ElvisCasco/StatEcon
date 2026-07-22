# ============================================================================
# stata_xtivreg.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    _c9_fe_iv_coefs(sub, idvar, yvar, xvars, endog, instruments)

Manual FE-IV 2SLS via within-transformation. Returns β in order of `xvars`.
Uses straight `inv` (no `pinv`) so weak-IV resamples return the true (possibly
extreme) estimates — needed for correct bootstrap SEs.
"""
function _c9_fe_iv_coefs(sub, idvar, yvar, xvars, endog, instruments)
    all_vars = unique(vcat([yvar], xvars, instruments))
    id_col = sub[!, idvar]
    id_to_rows = Dict{Any, Vector{Int}}()
    for (i, id) in enumerate(id_col)
        push!(get!(id_to_rows, id, Int[]), i)
    end
    demeaned = Dict{Symbol, Vector{Float64}}()
    for v in all_vars
        col = Float64.(sub[!, v])
        dc  = similar(col)
        for rows in values(id_to_rows)
            dc[rows] .= col[rows] .- Statistics.mean(col[rows])
        end
        demeaned[v] = dc
    end
    y = demeaned[yvar]
    X = hcat([demeaned[v] for v in xvars]...)
    Z_vars = vcat(filter(!=(endog), xvars), instruments)
    Z = hcat([demeaned[v] for v in Z_vars]...)

    ZtZi  = LinearAlgebra.inv(Z' * Z)
    P_Z   = Z * ZtZi * Z'
    XtPXi = LinearAlgebra.inv(X' * P_Z * X)
    return XtPXi * (X' * P_Z * y)
end

"""
    _c9_obs_bootstrap(f, df; B=400, seed=10101, verbose=true)

Observation-level bootstrap — resamples n rows with replacement (no cluster
structure). Matches Stata's `vce(bootstrap)` when no cluster is specified.
"""
function _c9_obs_bootstrap(f::Function, df; B::Int=400, seed::Int=10101, verbose::Bool=true)
    rng = Random.MersenneTwister(seed)
    n = DataFrames.nrow(df)
    results = Vector{Vector{Float64}}()
    n_fail = 0
    for b in 1:B
        row_idx = StatsBase.sample(rng, 1:n, n; replace=true)
        sub = df[row_idx, :]
        try
            push!(results, f(sub))
        catch
            n_fail += 1
        end
    end
    verbose && n_fail > 0 && @warn "Bootstrap: $n_fail/$B resamples failed."
    isempty(results) && error("All bootstrap resamples failed.")
    return reduce(hcat, results)'
end

"""
    _c9_panel_bootstrap(f, df, idvar; B=400, seed=10101, verbose=true)

Cluster bootstrap on the panel identifier `idvar`. Applies `f(df_resampled)` on
each resample and returns a matrix `B × k` of the vectors `f` returns.
"""
function _c9_panel_bootstrap(f::Function, df, idvar::Symbol; B::Int=400, seed::Int=10101,
                             verbose::Bool=true)
    rng = Random.MersenneTwister(seed)
    ids = unique(df[!, idvar])
    id_to_rows = Dict{Any, Vector{Int}}()
    for (i, id) in enumerate(df[!, idvar])
        push!(get!(id_to_rows, id, Int[]), i)
    end
    results = Vector{Vector{Float64}}()
    n_fail = 0
    for b in 1:B
        boot_ids = StatsBase.sample(rng, ids, length(ids); replace=true)
        row_idx = Int[]
        new_id  = Int[]
        for (k, i) in enumerate(boot_ids)
            rows = id_to_rows[i]
            append!(row_idx, rows)
            append!(new_id, fill(k, length(rows)))
        end
        sub = df[row_idx, :]
        sub[!, idvar] = new_id
        try
            push!(results, f(sub))
        catch
            n_fail += 1
        end
    end
    if verbose && n_fail > 0
        @warn "Bootstrap: $n_fail/$B resamples failed (skipped)."
    end
    isempty(results) && error("All bootstrap resamples failed.")
    return reduce(hcat, results)'
end

"""
    stata_xtivreg(df; y, xvars, endog, instruments, idvar, fe=true, vce=:bootstrap, reps, seed, level)

Stata-style `xtivreg y xvars (endog = instruments), fe vce(...)` — within-FE
2SLS with bootstrap standard errors. Prints a Stata-style table.
"""
function stata_xtivreg(df; y::Symbol, xvars::Vector{Symbol},
                 endog::Symbol, instruments::Vector{Symbol},
                 idvar::Symbol, fe::Bool=true,
                 vce::Symbol=:bootstrap, reps::Int=400, seed::Int=10101,
                 level::Float64=0.95)
    needed = unique(vcat([y, idvar], xvars, [endog], instruments))
    d = DataFrames.dropmissing(df[:, needed])
    for c in needed
        if eltype(d[!, c]) <: Union{Float32, Missing}
            d[!, c] = convert.(Union{Float64, Missing}, d[!, c])
        end
        eltype(d[!, c]) == Float32 && (d[!, c] = Float64.(d[!, c]))
    end

    all_regs = [endog; xvars...]      # β[1] is endog coef
    β = _c9_fe_iv_coefs(d, idvar, y, all_regs, endog, instruments)
    k = length(β)

    if vce == :bootstrap
        boot_β = _c9_obs_bootstrap(d; B=reps, seed=seed) do sub
            _c9_fe_iv_coefs(sub, idvar, y, all_regs, endog, instruments)
        end
    elseif vce == :bootstrap_cluster
        boot_β = _c9_panel_bootstrap(d, idvar; B=reps, seed=seed) do sub
            _c9_fe_iv_coefs(sub, idvar, y, all_regs, endog, instruments)
        end
    else
        error("vce = $vce not yet supported (use :bootstrap or :bootstrap_cluster)")
    end
    V  = Statistics.cov(boot_β)
    se = vec(Statistics.std(boot_β, dims=1))

    z    = β ./ se
    pv   = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    lo   = β .- crit .* se
    hi   = β .+ crit .* se

    id_col = d[!, idvar]
    panels = sort(unique(id_col))
    N = length(panels)
    n = DataFrames.nrow(d)

    id_to_rows = Dict{Any, Vector{Int}}()
    for (i, id) in enumerate(id_col)
        push!(get!(id_to_rows, id, Int[]), i)
    end
    T_counts = [length(id_to_rows[p]) for p in panels]
    T_avg = Statistics.mean(T_counts); T_min = minimum(T_counts); T_max = maximum(T_counts)

    ymeans = [Statistics.mean(Float64.(d[id_to_rows[p], y])) for p in panels]
    xmeans = Dict(v => [Statistics.mean(Float64.(d[id_to_rows[p], v])) for p in panels]
                  for v in all_regs)

    αhat = [ymeans[i] - sum(β[j] * xmeans[all_regs[j]][i] for j in 1:k)
            for i in 1:N]

    e_w = zeros(n)
    for (idx_p, p) in enumerate(panels)
        rows = id_to_rows[p]
        yw = Float64.(d[rows, y]) .- ymeans[idx_p]
        Xw = hcat([Float64.(d[rows, v]) .- xmeans[v][idx_p] for v in all_regs]...)
        e_w[rows] = yw - Xw * β
    end
    σ2_e = sum(e_w.^2) / (n - N - k)
    σ_e  = sqrt(σ2_e)

    σ2_u = max(Statistics.var(αhat) - σ2_e / T_avg, 0.0)
    σ_u  = sqrt(σ2_u)
    ρ    = σ2_u / (σ2_u + σ2_e)

    y_all  = Float64.(d[!, y])
    Xβ_all = hcat([Float64.(d[!, v]) for v in all_regs]...) * β
    r2_overall = Statistics.cor(y_all, Xβ_all)^2

    Xβ_panel = [sum(β[j] * xmeans[all_regs[j]][i] for j in 1:k) for i in 1:N]
    r2_between = Statistics.cor(ymeans, Xβ_panel)^2

    tss_w = 0.0
    for (idx_p, p) in enumerate(panels)
        rows = id_to_rows[p]
        tss_w += sum((Float64.(d[rows, y]) .- ymeans[idx_p]).^2)
    end
    rss_w = sum(e_w.^2)
    r2_within = tss_w > 0 ? 1 - rss_w / tss_w : NaN

    corr_ux = Statistics.cor(αhat, Xβ_panel)

    Wald   = β' * LinearAlgebra.inv(V) * β
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(k), Wald)

    df1, df2 = N - 1, n - N - k
    F_val = Statistics.var(αhat) * T_avg / σ2_e
    F_p   = 1 - Distributions.cdf(Distributions.FDist(df1, df2), F_val)

    ȳ = Statistics.mean(y_all)
    x̄ = [Statistics.mean(Float64.(d[!, v])) for v in all_regs]
    β0 = ȳ - LinearAlgebra.dot(β, x̄)
    β0_boot = [Statistics.mean(y_all) - sum(boot_β[b, j] * x̄[j] for j in 1:k)
               for b in 1:size(boot_β, 1)]
    se_cons = Statistics.std(β0_boot)
    z0   = β0 / se_cons
    p0   = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z0)))
    lo0  = β0 - crit * se_cons
    hi0  = β0 + crit * se_cons

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return lpad(s, w)
    end
    function commafmt(nn)
        s = string(abs(nn)); parts = String[]; i = length(s)
        while i >= 1
            push!(parts, s[max(1, i-2):i]); i -= 3
        end
        return (nn < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    println()
    Printf.@printf("%-48s%-18s= %10s\n",
                   "Fixed-effects (within) IV regression",
                   "Number of obs", commafmt(n))
    Printf.@printf("Group variable: %-32s%-18s= %10d\n",
                   string(idvar), "Number of groups", N)
    println()
    println("R-squared:                                      Obs per group:")
    Printf.@printf("     Within  = %-37s%-5s = %10d\n",
                   isnan(r2_within) ? "." : Printf.@sprintf("%.4f", r2_within),
                   "min", T_min)
    Printf.@printf("     Between = %-37s%-5s = %10.1f\n",
                   Printf.@sprintf("%.4f", r2_between), "avg", T_avg)
    Printf.@printf("     Overall = %-37s%-5s = %10d\n",
                   Printf.@sprintf("%.4f", r2_overall), "max", T_max)
    println()
    println()
    Printf.@printf("%48sWald chi2(%d)      = %10s\n",
                   "", k, Printf.@sprintf("%10.2f", Wald))
    Printf.@printf("corr(u_i, Xb) = %-32sProb > chi2       = %10.4f\n",
                   Printf.@sprintf("%.4f", corr_ux), Wald_p)
    println()
    if vce == :bootstrap_cluster
        Printf.@printf("%79s\n", "(Replications based on $N clusters in $idvar)")
    else
        Printf.@printf("%79s\n", "(Bootstrap replications = $reps)")
    end
    println("-"^78)
    println("             |   Observed   Bootstrap                         Normal-based")
    Printf.@printf("%12s | coefficient  std. err.      z    P>|z|     [%d%% conf. interval]\n",
                   string(y), round(Int, 100*level))
    println("-"^13, "+", "-"^64)
    for (i, v) in enumerate(all_regs)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       string(v),
                       g9(β[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", pv[i]),
                       g9(lo[i]; w=10), g9(hi[i]; w=10))
    end
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   "_cons",
                   g9(β0; w=10), g9(se_cons; w=9),
                   Printf.@sprintf("%6.2f", z0),
                   Printf.@sprintf("%.3f", p0),
                   g9(lo0; w=10), g9(hi0; w=10))
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s\n", "sigma_u", g9(σ_u; w=10))
    Printf.@printf("%12s | %s\n", "sigma_e", g9(σ_e; w=10))
    Printf.@printf("%12s | %s   (fraction of variance due to u_i)\n",
                   "rho", g9(ρ; w=10))
    println("-"^78)
    Printf.@printf(" F test that all u_i=0: F(%d,%d) = %8.2f            Prob > F    = %.4f\n",
                   df1, df2, F_val, F_p)
    println("-"^78)
    println("Endogenous: ", endog)
    println("Exogenous:  ", join(xvars, " "), " ", join(instruments, " "))

    return (; β, se, z, pv, ci_lo=lo, ci_hi=hi, β0, se_cons,
              σ_u, σ_e, ρ, r2_within, r2_between, r2_overall,
              corr_ux, Wald, Wald_p, F_val, F_p,
              nobs=n, N_panels=N, T_min, T_avg, T_max,
              coefnames=string.(all_regs))
end
