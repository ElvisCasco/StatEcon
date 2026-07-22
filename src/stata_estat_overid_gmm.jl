# ============================================================================
# stata_estat_overid_gmm.jl — Stata `estat overid` after GMM (C&T ch17)
# ============================================================================

"""
    stata_estat_overid_gmm(gmm_res) -> NamedTuple

Stata's `estat overid` after `gmm` — Hansen's J test of overidentifying
restrictions, read from a `stata_poisson_gmm(...; twostep=true)` result.
Prints `Hansen's J chi2(df) = …  (p = …)`. (Valid only after the
two-step fit, where the weight matrix is the optimal Ŝ⁻¹.)

Returns `(; J, df, p)`.
"""
function stata_estat_overid_gmm(gmm_res)
    gmm_res.twostep ||
        @warn "estat overid: J is the efficient-GMM statistic only after the two-step fit."
    println()
    Printf.@printf("Test of overidentifying restriction:\n")
    Printf.@printf("Hansen's J chi2(%d) = %.5f  (p = %.4f)\n",
                   gmm_res.J_df, gmm_res.Jstat, gmm_res.J_p)
    return (; J = gmm_res.Jstat, df = gmm_res.J_df, p = gmm_res.J_p)
end
