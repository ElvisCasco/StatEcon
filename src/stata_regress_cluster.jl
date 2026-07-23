# Weighted OLS with one-way cluster-robust standard errors, matching Stata's
# `regress y x1 x2 [pweight=w], vce(cluster id)`. Uses the hc1-style
# adjustment (M/(M-1)) * (N-1)/(N-k) on the meat matrix, t-tests on M-1 df,
# model F distributed F(q, M-1), Root MSE with pweights normalized to sum
# to N (Stata convention).
#
#   stata_regress_cluster(df, :hgb, [:age, :female];
#                         cluster = :uniqpsu, weight = :finalwgt)
#
# `weight=nothing` runs unweighted OLS with cluster-robust SEs.

function stata_regress_cluster(df, y, xs::AbstractVector;
                               cluster::Symbol,
                               weight::Union{Symbol,Nothing} = nothing,
                               level::Float64 = 0.95)
    ys  = Symbol(y)
    xsv = [Symbol(v) for v in xs]
    needed = unique(vcat(ys, xsv, cluster))
    weight !== nothing && push!(needed, weight)
    d = DataFrames.dropmissing(df, needed)
    Y = Float64.(_sm_rawval.(d[!, ys]))
    Xcols = [Float64.(_sm_rawval.(d[!, v])) for v in xsv]
    X = hcat(ones(length(Y)), Xcols...)
    c = _sm_rawval.(d[!, cluster])
    w = weight === nothing ? ones(length(Y)) :
        Float64.(_sm_rawval.(d[!, weight]))
    cn = vcat("_cons", string.(xsv))

    n = length(Y); k = size(X, 2); q = k - 1

    XW = X .* w
    A  = XW' * X
    β  = A \ (XW' * Y)
    e  = Y .- X * β
    U  = XW .* e

    clusters = unique(c)
    M = length(clusters)
    meat = zeros(k, k)
    for cl in clusters
        sel = c .== cl
        s = sum(U[sel, :]; dims = 1)
        meat .+= s' * s
    end
    adj   = (M / (M - 1)) * ((n - 1) / (n - k))
    A_inv = LinearAlgebra.inv(A)
    V     = adj .* (A_inv * meat * A_inv)
    se    = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    dfree  = M - 1
    t_stat = β ./ se
    tcrit  = Distributions.quantile(Distributions.TDist(dfree), 1 - (1 - level) / 2)
    pvals  = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(dfree), abs.(t_stat)))
    ci_lo  = β .- tcrit .* se
    ci_hi  = β .+ tcrit .* se

    slope = 2:k
    Wstat = β[slope]' * (V[slope, slope] \ β[slope])
    Fstat = Wstat / q
    pF    = 1 - Distributions.cdf(Distributions.FDist(q, dfree), Fstat)

    ybar_w = sum(w .* Y) / sum(w)
    rss_w  = sum(w .* e .^ 2)
    tss_w  = sum(w .* (Y .- ybar_w) .^ 2)
    r2     = 1 - rss_w / tss_w

    w_norm = w .* (n / sum(w))
    rmse   = sqrt(sum(w_norm .* e .^ 2) / (n - k))

    Printf.@printf("%-48s%-18s= %10s\n",  "Linear regression", "Number of obs", _sm_comma(n))
    Printf.@printf("%-48s%-18s= %10.2f\n", "", "F($q, $dfree)", Fstat)
    Printf.@printf("%-48s%-18s= %10.4f\n", "", "Prob > F", pF)
    Printf.@printf("%-48s%-18s= %10.4f\n", "", "R-squared", r2)
    Printf.@printf("%-48s%-18s= %10.4f\n", "", "Root MSE", rmse)
    println()
    Printf.@printf("%78s\n", "(Std. err. adjusted for $M clusters in $cluster)")

    order = vcat(collect(slope), 1)
    lvl = round(Int, 100 * level)
    headers = [string(ys), "Coefficient", "Robust SE",
               "t", "P>|t|", "[$(lvl)% CI low]", "[$(lvl)% CI high]"]
    rows = [[cn[i],
             Printf.@sprintf("%.7f", β[i]),
             Printf.@sprintf("%.7f", se[i]),
             Printf.@sprintf("%.2f", t_stat[i]),
             Printf.@sprintf("%.3f", pvals[i]),
             Printf.@sprintf("%.7f", ci_lo[i]),
             Printf.@sprintf("%.7f", ci_hi[i])] for i in order]
    _sug_print_table(headers, rows, [:l, :r, :r, :r, :r, :r, :r])

    return (; β, V, se, t = t_stat, p = pvals, ci_lo, ci_hi, coefnames = cn,
              n_obs = n, M, df = dfree,
              F = Fstat, df1 = q, df2 = dfree, pF, r2, rmse)
end
