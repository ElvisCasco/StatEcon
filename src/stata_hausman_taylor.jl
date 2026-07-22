# ============================================================================
# stata_hausman_taylor.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_hausman_taylor(df, y, idvar; time_exog, time_endog, indiv_exog, indiv_endog, vce, reps, seed)

Stata-style `xthtaylor` — Hausman-Taylor (1981) estimator for panel data with
time-invariant regressors. `vce` may be `:default`, `:bootstrap`, or `:jackknife`.
"""
function stata_hausman_taylor(df, y::Symbol, idvar::Symbol;
                        time_exog::Vector{Symbol}   = Symbol[],
                        time_endog::Vector{Symbol}  = Symbol[],
                        indiv_exog::Vector{Symbol}  = Symbol[],
                        indiv_endog::Vector{Symbol} = Symbol[],
                        vce::Symbol = :default,
                        reps::Int = 400,
                        seed::Int = 10101,
                        _ht_coefs_only::Bool = false)
    all_X = vcat(time_exog, time_endog, indiv_exog, indiv_endog)
    needed = unique(vcat([y, idvar], all_X))
    d = DataFrames.dropmissing(df, needed)
    n = DataFrames.nrow(d)
    gdf = DataFrames.groupby(d, idvar)
    N   = length(gdf)
    time_all = vcat(time_exog, time_endog)
    k_tv = length(time_all)

    means_df = DataFrames.combine(gdf,
        (v => (x -> Statistics.mean(Float64.(_c9_rawval.(collect(skipmissing(x)))))) =>
            Symbol(v, :_bar) for v in vcat([y], all_X))...)
    d2 = DataFrames.leftjoin(d, means_df, on = idvar)

    ỹ = Float64.(d2[!, y]) .- d2[!, Symbol(y, :_bar)]
    X̃_tv = hcat([Float64.(d2[!, v]) .- d2[!, Symbol(v, :_bar)] for v in time_all]...)
    β_fe = X̃_tv \ ỹ
    e_fe = ỹ .- X̃_tv * β_fe
    T_bar = n / N
    σ_ε2 = sum(abs2, e_fe) / (n - N)

    fe_i = Vector{Float64}(undef, N)
    for (j, g) in enumerate(gdf)
        ȳ = Statistics.mean(Float64.(_c9_rawval.(g[!, y])))
        x̄ = [Statistics.mean(Float64.(_c9_rawval.(g[!, v]))) for v in time_all]
        fe_i[j] = ȳ - LinearAlgebra.dot(x̄, β_fe)
    end

    indiv_all = vcat(indiv_exog, indiv_endog)
    ti_mat = Matrix{Float64}(undef, N, length(indiv_all))
    for (j, g) in enumerate(gdf)
        for (c, v) in enumerate(indiv_all)
            ti_mat[j, c] = Float64(_c9_rawval(first(g[!, v])))
        end
    end

    ti_exog_mat = Matrix{Float64}(undef, N, length(indiv_exog))
    for (j, g) in enumerate(gdf)
        for (c, v) in enumerate(indiv_exog)
            ti_exog_mat[j, c] = Float64(_c9_rawval(first(g[!, v])))
        end
    end

    W_fe = Matrix{Float64}(undef, N, length(time_exog))
    for (j, g) in enumerate(gdf)
        for (c, v) in enumerate(time_exog)
            W_fe[j, c] = Statistics.mean(Float64.(_c9_rawval.(g[!, v])))
        end
    end

    ti_mat_c = hcat(ones(N), ti_mat)
    W_fe_c   = hcat(ones(N), ti_exog_mat, W_fe)
    P_W = W_fe_c * ((W_fe_c'W_fe_c) \ W_fe_c')
    β_ti_full = (ti_mat_c' * P_W * ti_mat_c) \ (ti_mat_c' * P_W * fe_i)
    e_ti = fe_i .- ti_mat_c * β_ti_full
    s21 = sum(abs2, e_ti) / N
    σ_u2 = max(s21 - σ_ε2 / T_bar, 0.0)

    θ = 1 - sqrt(σ_ε2 / (T_bar * σ_u2 + σ_ε2))
    for v in vcat([y], all_X)
        d2[!, Symbol(v, :_star)] = Float64.(d2[!, v]) .- θ .* d2[!, Symbol(v, :_bar)]
    end

    for v in time_all
        d2[!, Symbol(v, :_w)] = Float64.(d2[!, v]) .- d2[!, Symbol(v, :_bar)]
    end
    for v in time_exog
        d2[!, Symbol(v, :_bbar)] = d2[!, Symbol(v, :_bar)]
    end

    X_cols = Symbol.(string.(all_X) .* "_star")
    X = hcat(fill(1 - θ, n), Matrix{Float64}(d2[:, X_cols]))
    y_star = Vector{Float64}(d2[!, Symbol(y, :_star)])

    Z_parts = Vector{Matrix{Float64}}()
    push!(Z_parts, fill(1 - θ, n, 1))
    push!(Z_parts, hcat([Vector{Float64}(d2[!, Symbol(v, :_w)]) for v in time_all]...))
    if !isempty(indiv_exog)
        push!(Z_parts, hcat([Float64.(d2[!, v]) .- θ .* d2[!, Symbol(v, :_bar)]
                             for v in indiv_exog]...))
    end
    if !isempty(time_exog)
        push!(Z_parts, hcat([Vector{Float64}(d2[!, Symbol(v, :_bar)]) for v in time_exog]...))
    end
    Z = hcat(Z_parts...)

    P_Z = Z * ((Z'Z) \ Z')
    XPZ = X' * P_Z
    β = (XPZ * X) \ (XPZ * y_star)
    V = σ_ε2 .* LinearAlgebra.inv(XPZ * X)

    _ht_coefs_only && return β

    panel_ids = unique(d[!, idvar])
    id_to_rows = Dict{Any, Vector{Int}}()
    for (i, id) in enumerate(d[!, idvar])
        push!(get!(id_to_rows, id, Int[]), i)
    end

    if vce == :bootstrap
        rng = Random.MersenneTwister(seed)
        boot_β = Matrix{Float64}(undef, 0, length(β))
        n_fail = 0
        for b in 1:reps
            boot_ids = StatsBase.sample(rng, panel_ids, length(panel_ids); replace=true)
            row_idx = Int[]; new_id = Int[]
            for (k, idv) in enumerate(boot_ids)
                rows = id_to_rows[idv]
                append!(row_idx, rows)
                append!(new_id, fill(k, length(rows)))
            end
            sub = d[row_idx, :]
            sub[!, idvar] = new_id
            try
                β_b = stata_hausman_taylor(sub, y, idvar;
                    time_exog=time_exog, time_endog=time_endog,
                    indiv_exog=indiv_exog, indiv_endog=indiv_endog,
                    _ht_coefs_only=true)
                boot_β = vcat(boot_β, β_b')
            catch
                n_fail += 1
            end
        end
        n_fail > 0 && @warn "Bootstrap: $n_fail/$reps resamples failed."
        V  = Statistics.cov(boot_β)
        se = vec(Statistics.std(boot_β, dims=1))
    elseif vce == :jackknife
        nβ = length(β)
        jack_β = Matrix{Float64}(undef, 0, nβ)
        n_fail = 0
        for idv in panel_ids
            sub = d[d[!, idvar] .!= idv, :]
            try
                β_j = stata_hausman_taylor(sub, y, idvar;
                    time_exog=time_exog, time_endog=time_endog,
                    indiv_exog=indiv_exog, indiv_endog=indiv_endog,
                    _ht_coefs_only=true)
                jack_β = vcat(jack_β, β_j')
            catch
                n_fail += 1
            end
        end
        n_fail > 0 && @warn "Jackknife: $n_fail/$N panels produced errors."
        N_used = size(jack_β, 1)
        β_bar  = vec(Statistics.mean(jack_β, dims=1))
        dev    = jack_β .- β_bar'
        V      = ((N_used - 1) / N_used) .* (dev' * dev)
        se     = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    else
        se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    end

    use_t = (vce == :jackknife)
    if use_t
        df_t = N - 1
        z  = β ./ se
        p  = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(df_t), abs.(z)))
    else
        z  = β ./ se
        p  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    end

    colnames = vcat(["_cons"], string.(all_X))

    σ_u = sqrt(σ_u2);  σ_e = sqrt(σ_ε2)
    ρ   = σ_u2 / (σ_u2 + σ_ε2)

    β_slope = β[2:end]
    V_slope = V[2:end, 2:end]
    Wald_chi2 = β_slope' * LinearAlgebra.inv(V_slope) * β_slope
    k_slope = length(β_slope)
    if use_t
        Wald = Wald_chi2 / k_slope
        Wald_p = 1 - Distributions.cdf(Distributions.FDist(k_slope, N - 1), Wald)
    else
        Wald = Wald_chi2
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(k_slope), Wald)
    end

    T_counts = [DataFrames.nrow(g) for g in gdf]
    T_min = minimum(T_counts); T_max_v = maximum(T_counts)

    crit = use_t ? Distributions.quantile(Distributions.TDist(N - 1), 0.975) : 1.96
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

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

    idx_of = Dict{Symbol,Int}()
    pos = 2
    for v in time_exog;   idx_of[v] = pos; pos += 1; end
    for v in time_endog;  idx_of[v] = pos; pos += 1; end
    for v in indiv_exog;  idx_of[v] = pos; pos += 1; end
    for v in indiv_endog; idx_of[v] = pos; pos += 1; end

    println()
    Printf.@printf("%-48s%-18s= %10s\n",
                   "Hausman-Taylor estimation", "Number of obs", commafmt(n))
    Printf.@printf("Group variable: %-32s%-18s= %10d\n",
                   string(idvar), "Number of groups", N)
    println()
    println("                                                Obs per group:")
    Printf.@printf("%48s%-5s = %10d\n", "", "min", T_min)
    Printf.@printf("%48s%-5s = %10d\n", "", "avg", round(Int, T_bar))
    Printf.@printf("%48s%-5s = %10d\n", "", "max", T_max_v)
    println()
    if use_t
        Printf.@printf("Random effects u_i ~ i.i.d.%21sF(%3d, %6d)   = %10.2f\n",
                       "", k_slope, N - 1, Wald)
        Printf.@printf("%48sProb > F          = %10.4f\n", "", Wald_p)
    else
        Printf.@printf("Random effects u_i ~ i.i.d.%21sWald chi2(%d)     = %10.2f\n",
                       "", k_slope, Wald)
        Printf.@printf("%48sProb > chi2       = %10.4f\n", "", Wald_p)
    end
    println()
    if vce == :bootstrap
        Printf.@printf("%79s\n", "(Replications based on $N clusters in $idvar)")
    elseif vce == :jackknife
        Printf.@printf("%79s\n", "(Replications based on $N clusters in $idvar)")
    end
    println("-"^78)
    if vce == :bootstrap
        println("             |   Observed   Bootstrap                         Normal-based")
        Printf.@printf("%12s | coefficient  std. err.      z    P>|z|     [95%% conf. interval]\n",
                       string(y))
    elseif vce == :jackknife
        println("             |              Jackknife")
        Printf.@printf("%12s | Coefficient  std. err.      t    P>|t|     [95%% conf. interval]\n",
                       string(y))
    else
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [95%% conf. interval]\n",
                       string(y))
    end
    println("-"^13, "+", "-"^64)

    function prow(v::Symbol)
        i = idx_of[v]
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       string(v),
                       g9(β[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       g9(ci_lo[i]; w=10), g9(ci_hi[i]; w=10))
    end

    if !isempty(time_exog)
        println("TVexogenous  |")
        for v in time_exog;  prow(v);  end
    end
    if !isempty(time_endog)
        println("TVendogenous |")
        for v in time_endog; prow(v);  end
    end
    if !isempty(indiv_exog)
        println("TIexogenous  |")
        for v in indiv_exog; prow(v);  end
    end
    if !isempty(indiv_endog)
        println("TIendogenous |")
        for v in indiv_endog; prow(v); end
    end
    println("             |")
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   "_cons",
                   g9(β[1]; w=10), g9(se[1]; w=9),
                   Printf.@sprintf("%6.2f", z[1]),
                   Printf.@sprintf("%.3f", p[1]),
                   g9(ci_lo[1]; w=10), g9(ci_hi[1]; w=10))
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s\n", "sigma_u", g9(σ_u; w=10))
    Printf.@printf("%12s | %s\n", "sigma_e", g9(σ_e; w=10))
    Printf.@printf("%12s | %s   (fraction of variance due to u_i)\n",
                   "rho", g9(ρ; w=10))
    println("-"^78)
    println("Note: TV refers to time varying; TI refers to time invariant.")

    return (; β, V, se, z, p, ci_lo, ci_hi,
              coefnames = colnames, σ_ε2, σ_u2, σ_u, σ_e, ρ, θ, n,
              Wald, Wald_p, nobs=n, N_panels=N,
              T_min, T_avg=round(Int, T_bar), T_max=T_max_v)
end
