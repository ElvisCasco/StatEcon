# ============================================================================
# stata_estimates_stats.jl — Stata `estimates stats` AIC/BIC (C&T ch17)
# ============================================================================

"""
    stata_estimates_stats(models) -> NamedTuple

Stata's `estimates stats <m1> <m2> …` — side-by-side AIC/BIC comparison.
`models` is a vector of tuples `(name, n, ll, df)` or, to fill the
`ll(null)` column (each command's `e(ll_0)`, the intercept-only fit's
log-likelihood), `(name, n, ll, df, ll_null)`. A missing/`NaN` ll_null
prints as ".". Per row:

    AIC = −2·ll + 2·df,   BIC = −2·ll + df·log(N).

Prints Stata's wide table; returns `(; aic, bic)` vectors in input order.
"""
function stata_estimates_stats(models::AbstractVector)
    commafmt(num::Integer) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end
    aics = Float64[]; bics = Float64[]
    println("\nAkaike's information criterion and Bayesian information criterion\n")
    println("-"^78)
    println("       Model |          N   ll(null)  ll(model)      df        AIC        BIC")
    println("-"^13, "+", "-"^64)
    for row in models
        name, n, ll, df = row[1], row[2], row[3], row[4]
        ll0 = length(row) >= 5 ? row[5] : NaN
        aic = -2*ll + 2*df
        bic = -2*ll + df*log(n)
        push!(aics, aic); push!(bics, bic)
        ll0_str = (ismissing(ll0) || !isfinite(ll0)) ?
                  "." : Printf.@sprintf("%.6g", ll0)
        Printf.@printf("%12s | %10s %10s %10.7g %7d %10.7g %10.7g\n",
                       string(name), commafmt(n), ll0_str, ll, df, aic, bic)
    end
    println("-"^78)
    println("Note: BIC uses N = number of observations. See [R] IC note.")
    return (; aic = aics, bic = bics)
end
