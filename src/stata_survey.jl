# Survey mean / regression with Taylor-linearized (Horvitz-Thompson)
# cluster-robust variance under stratified single-stage cluster designs.
#
#   stata_mean(df, :hgb)                              # `mean hgb`
#   stata_svy_mean(df, :hgb; stratum=:strata, psu=:psu, weight=:finalwgt)
#   stata_svy_regress(df, :hgb, [:age, :female];
#                     stratum=:strata, psu=:psu, weight=:finalwgt)

# Table helpers shared with stata_sureg / stata_regress_cluster (via _sug_*).

# --- stata_mean: plain sample mean with t-based CI on n-1 df --------------
function stata_mean(df, var; level::Float64 = 0.95)
    vs = Symbol(var)
    d  = DataFrames.dropmissing(df, [vs])
    y  = Float64.(_sm_rawval.(d[!, vs]))
    n  = length(y)
    μ  = Statistics.mean(y)
    s  = Statistics.std(y)
    se = s / sqrt(n)
    dfree = n - 1
    tcrit = Distributions.quantile(Distributions.TDist(dfree), 1 - (1 - level) / 2)
    ci_lo = μ - tcrit * se
    ci_hi = μ + tcrit * se

    @printf("Mean estimation                          Number of obs = %s\n\n",
            _sm_comma(n))
    lvl = round(Int, 100 * level)
    _sug_print_table(
        ["Variable", "Mean", "Std. err.", "[$(lvl)% CI low]", "[$(lvl)% CI high]"],
        [[string(var),
          @sprintf("%.5f", μ), @sprintf("%.5f", se),
          @sprintf("%.5f", ci_lo), @sprintf("%.5f", ci_hi)]],
        [:l, :r, :r, :r, :r])
    return (; mean = μ, se = se, ci = (ci_lo, ci_hi), df = dfree, n_obs = n)
end

# --- stata_svy_mean: weighted mean with linearized cluster-stratified SE --
function stata_svy_mean(df, var; stratum::Symbol, psu::Symbol, weight::Symbol,
                        level::Float64 = 0.95)
    d = DataFrames.dropmissing(df, [Symbol(var), stratum, psu, weight])
    y = Float64.(_sm_rawval.(d[!, Symbol(var)]))
    w = Float64.(_sm_rawval.(d[!, weight]))
    h = _sm_rawval.(d[!, stratum])
    c = _sm_rawval.(d[!, psu])

    n = length(y); W = sum(w)
    ybar = sum(w .* y) / W
    z = w .* (y .- ybar) ./ W               # Taylor-linearized score

    V = 0.0
    strata = unique(h)
    n_psu = 0
    for hv in strata
        in_h = h .== hv
        psus_h = c[in_h]
        z_h = z[in_h]
        ups = unique(psus_h)
        nh = length(ups)
        n_psu += nh
        nh < 2 && continue
        u = [sum(z_h[psus_h .== p]) for p in ups]
        ū = Statistics.mean(u)
        V += (nh / (nh - 1)) * sum((u .- ū) .^ 2)
    end
    se = sqrt(V)
    n_str = length(strata)
    dfree = n_psu - n_str
    tcrit = Distributions.quantile(Distributions.TDist(dfree), 1 - (1 - level) / 2)
    ci_lo = ybar - tcrit * se
    ci_hi = ybar + tcrit * se

    println("Survey: Mean estimation\n")
    @printf("Number of strata = %-13d Number of obs   = %11s\n",
            n_str, _sm_comma(n))
    @printf("Number of PSUs   = %-13d Population size = %11s\n",
            n_psu, _sm_comma(Int(round(W))))
    @printf("%33s Design df       = %11d\n", "", dfree)
    println()

    lvl = round(Int, 100 * level)
    _sug_print_table(
        ["Variable", "Mean", "Linearized SE", "[$(lvl)% CI low]", "[$(lvl)% CI high]"],
        [[string(var),
          @sprintf("%.5f", ybar), @sprintf("%.5f", se),
          @sprintf("%.5f", ci_lo), @sprintf("%.5f", ci_hi)]],
        [:l, :r, :r, :r, :r])
    return (; mean = ybar, se = se, ci = (ci_lo, ci_hi), df = dfree,
              n_obs = n, n_strata = n_str, n_psu, popsize = W)
end

# --- stata_svy_regress: weighted OLS with cluster-stratified linearized VCV
function stata_svy_regress(df, y, xs::AbstractVector;
                           stratum::Symbol, psu::Symbol, weight::Symbol,
                           level::Float64 = 0.95)
    ys = Symbol(y)
    xsv = [Symbol(v) for v in xs]
    d = DataFrames.dropmissing(df, unique([ys, xsv..., stratum, psu, weight]))
    Y = Float64.(_sm_rawval.(d[!, ys]))
    w = Float64.(_sm_rawval.(d[!, weight]))
    h = _sm_rawval.(d[!, stratum])
    c = _sm_rawval.(d[!, psu])
    Xcols = [Float64.(_sm_rawval.(d[!, v])) for v in xsv]
    X = hcat(ones(length(Y)), Xcols...)
    cn = vcat("_cons", string.(xsv))

    n = length(Y); k = size(X, 2); q = k - 1

    XW = X .* w
    A = XW' * X
    β = A \ (XW' * Y)
    e = Y .- X * β
    U = XW .* e                          # scores uᵢ = wᵢ xᵢ eᵢ

    strata = unique(h)
    V_meat = zeros(k, k)
    n_psu = 0
    for hv in strata
        in_h = h .== hv
        psus = c[in_h]
        U_h = U[in_h, :]
        ups = unique(psus)
        nh = length(ups)
        n_psu += nh
        nh < 2 && continue
        S = reduce(vcat, [sum(U_h[psus .== p, :]; dims = 1) for p in ups])
        S̄ = Statistics.mean(S; dims = 1)
        centered = S .- S̄
        V_meat .+= (nh / (nh - 1)) .* (centered' * centered)
    end
    A_inv = LinearAlgebra.inv(A)
    V = A_inv * V_meat * A_inv
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    n_str = length(strata)
    dfree = n_psu - n_str
    t_stat = β ./ se
    tcrit = Distributions.quantile(Distributions.TDist(dfree), 1 - (1 - level) / 2)
    pvals = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(dfree), abs.(t_stat)))
    ci_lo = β .- tcrit .* se
    ci_hi = β .+ tcrit .* se

    slope = 2:k
    Wstat = β[slope]' * (V[slope, slope] \ β[slope])
    Fstat = Wstat / q
    df2 = dfree - q + 1
    pF = 1 - Distributions.cdf(Distributions.FDist(q, df2), Fstat)

    ybar_w = sum(w .* Y) / sum(w)
    rss_w = sum(w .* e .^ 2)
    tss_w = sum(w .* (Y .- ybar_w) .^ 2)
    r2 = 1 - rss_w / tss_w
    W_tot = sum(w)

    println("Survey: Linear regression\n")
    @printf("Number of strata = %-30dNumber of obs   = %11s\n", n_str, _sm_comma(n))
    @printf("Number of PSUs   = %-30dPopulation size = %11s\n", n_psu, _sm_comma(Int(round(W_tot))))
    @printf("%-49sDesign df       = %11d\n", "", dfree)
    @printf("%-49s%-16s= %11.2f\n", "", "F($q, $df2)", Fstat)
    @printf("%-49s%-16s= %11.4f\n", "", "Prob > F", pF)
    @printf("%-49s%-16s= %11.4f\n", "", "R-squared", r2)
    println()

    order = vcat(collect(slope), 1)
    lvl = round(Int, 100 * level)
    headers = [string(ys), "Coefficient", "Linearized SE",
               "t", "P>|t|", "[$(lvl)% CI low]", "[$(lvl)% CI high]"]
    rows = [[cn[i],
             @sprintf("%.7f", β[i]),
             @sprintf("%.7f", se[i]),
             @sprintf("%.2f", t_stat[i]),
             @sprintf("%.3f", pvals[i]),
             @sprintf("%.7f", ci_lo[i]),
             @sprintf("%.7f", ci_hi[i])] for i in order]
    _sug_print_table(headers, rows, [:l, :r, :r, :r, :r, :r, :r])

    return (; β, V, se, t = t_stat, p = pvals, ci_lo, ci_hi,
              coefnames = cn, n_obs = n, n_strata = n_str, n_psu,
              popsize = W_tot, df = dfree,
              F = Fstat, df1 = q, df2, pF, r2)
end
