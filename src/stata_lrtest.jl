# ============================================================================
# stata_lrtest.jl — Stata `lrtest` (Cameron & Trivedi ch12: Testing methods)
# ============================================================================

"""
    stata_lrtest(ll_unrestricted, ll_restricted, df) -> (; chi2, df, p)

Stata-style `lrtest` for two nested ML models. Computes
`LR = 2·(ll_unrestricted − ll_restricted)`, reports `LR chi2(df)` and
`Prob > chi2`, with Stata's "Assumption: restrict nested within unrestrict"
header line.

Use after fitting two ML models (e.g. two `stata_nbreg` calls with different
formulas) and pass their `.ll` fields. `df` is the number of restrictions
(regressors dropped between the unrestricted and restricted models).

Returns `(; chi2, df, p)`.
"""
function stata_lrtest(ll_unrestricted::Real, ll_restricted::Real, df::Int)
    chi2 = 2 * (Float64(ll_unrestricted) - Float64(ll_restricted))
    p    = Distributions.ccdf(Distributions.Chisq(df), chi2)
    println("Likelihood-ratio test")
    println("Assumption: restrict nested within unrestrict")
    println()
    Printf.@printf(" LR chi2(%d) = %.2f\n", df, chi2)
    Printf.@printf("Prob > chi2 = %.4f\n", p)
    return (; chi2, df, p)
end
