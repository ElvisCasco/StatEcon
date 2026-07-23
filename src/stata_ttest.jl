"""
    stata_ttest(df, var, by; level=0.95, unequal=false)

Stata-style `ttest <var>, by(<by>)`. Two-sample t-test between the two
non-missing levels of `by`. Defaults to equal-variance (pooled), matching
Stata; pass `unequal=true` for Welch's test.

Prints the two-group summary table, the difference with its confidence
interval, and one-/two-sided p-values. Returns a NamedTuple with fields
`(t, df, xbar, stderr, p, ci)`.
"""
function stata_ttest(df::AbstractDataFrame, var::Symbol, by::Symbol;
                     level::Float64 = 0.95, unequal::Bool = false)
    mask = .!ismissing.(df[!, var]) .& .!ismissing.(df[!, by])
    yvec = df[mask, var]
    bvec = [_sm_rawval(v) for v in df[mask, by]]
    levs = sort(unique(bvec))
    length(levs) == 2 ||
        error("by() must have exactly 2 non-missing levels, got $(length(levs))")

    g0 = Float64.(yvec[bvec .== levs[1]])
    g1 = Float64.(yvec[bvec .== levs[2]])

    n0, n1 = length(g0), length(g1)
    μ0, μ1 = Statistics.mean(g0), Statistics.mean(g1)
    s0, s1 = Statistics.std(g0), Statistics.std(g1)
    dif    = μ0 - μ1

    if unequal
        se  = sqrt(s0^2 / n0 + s1^2 / n1)
        dfr = (s0^2 / n0 + s1^2 / n1)^2 /
              ((s0^2 / n0)^2 / (n0 - 1) + (s1^2 / n1)^2 / (n1 - 1))
    else
        sp2 = ((n0 - 1) * s0^2 + (n1 - 1) * s1^2) / (n0 + n1 - 2)
        se  = sqrt(sp2 * (1 / n0 + 1 / n1))
        dfr = float(n0 + n1 - 2)
    end
    tstat = dif / se

    tdist = Distributions.TDist(max(dfr, 1))
    tc    = Distributions.quantile(tdist, 1 - (1 - level) / 2)
    plt   = Distributions.cdf(tdist, tstat)
    ptwo  = 2 * min(plt, 1 - plt)
    pgt   = 1 - plt
    ci    = (dif - tc * se, dif + tc * se)

    function _row(x, lbl)
        n   = length(x)
        μ   = Statistics.mean(x)
        σ   = n > 1 ? Statistics.std(x) : NaN
        sem = σ / sqrt(n)
        td  = Distributions.TDist(max(n - 1, 1))
        tcr = Distributions.quantile(td, 1 - (1 - level) / 2)
        (Group = string(lbl), Obs = n, Mean = μ, StdErr = sem, StdDev = σ,
         CI_lo = μ - tcr * sem, CI_hi = μ + tcr * sem)
    end

    println(unequal ? "Two-sample t test with unequal variances" :
                      "Two-sample t test with equal variances")
    println()
    Printf.@printf("%-9s %6s %11s %11s %11s   [%d%% conf. interval]\n",
                   "Group", "Obs", "Mean", "Std. err.", "Std. dev.",
                   round(Int, 100 * level))
    println("-"^80)
    for r in (_row(g0, levs[1]), _row(g1, levs[2]),
              _row(vcat(g0, g1), "combined"))
        Printf.@printf("%-9s %6d %11.5f %11.5f %11.5f  %10.5f  %10.5f\n",
                       r.Group, r.Obs, r.Mean, r.StdErr, r.StdDev,
                       r.CI_lo, r.CI_hi)
    end
    Printf.@printf("%-9s %6d %11.5f %11.5f %11s  %10.5f  %10.5f\n",
                   "diff", n0 + n1, dif, se, "", ci[1], ci[2])
    println()
    Printf.@printf("diff = mean(%s) - mean(%s)      t = %.4f\n",
                   levs[1], levs[2], tstat)
    Printf.@printf("Ho: diff = 0                        df = %g\n\n", dfr)
    println("    Ha: diff < 0         Ha: diff != 0         Ha: diff > 0")
    Printf.@printf("Pr(T < t) = %.4f   Pr(|T| > |t|) = %.4f    Pr(T > t) = %.4f\n",
                   plt, ptwo, pgt)

    return (t = tstat, df = dfr, xbar = dif, stderr = se, p = ptwo, ci = ci)
end
