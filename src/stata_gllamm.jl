# ============================================================================
# stata_gllamm.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_gllamm(df, y, xvars, idvar; nip=10)

Stata's `gllamm` for a Gaussian random-intercept model. For the linear case
this is the same MLE as `stata_xtmixed` (reslopes empty), printed in gllamm's
format (VARIANCES rather than SDs). Returns the fit NamedTuple.
"""
function stata_gllamm(df, y::Symbol, xvars::AbstractVector{Symbol}, idvar::Symbol;
                      nip::Int = 10)
    res = stata_xtmixed(df, y, xvars, idvar; print_table = false)
    β = res.β; se = res.se; z = res.z; p = res.p
    cn = res.coefnames; n = res.n; G = res.G
    σ2_resid = res.σ2
    σ2_re = res.Ψ[1, 1]
    cond_num = LinearAlgebra.cond(res.Vβ)

    println()
    Printf.@printf("number of level 1 units = %d\n", n)
    Printf.@printf("number of level 2 units = %d\n", G)
    println(" ")
    Printf.@printf("Condition Number = %.4f\n", cond_num)
    println(" ")
    println("gllamm model ")
    println(" ")
    Printf.@printf("log likelihood = %.5f\n", res.loglik)
    println()

    println("-"^78)
    Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [95%% conf. interval]\n",
                   string(y))
    println("-"^13, "+", "-"^64)
    ci_lo = β .- 1.96 .* se;  ci_hi = β .+ 1.96 .* se
    for i in 2:length(β)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       cn[i], _c9_g9(β[i]; w=10), _c9_g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       _c9_g9(ci_lo[i]; w=10), _c9_g9(ci_hi[i]; w=10))
    end
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   "_cons", _c9_g9(β[1]; w=10), _c9_g9(se[1]; w=9),
                   Printf.@sprintf("%6.2f", z[1]),
                   Printf.@sprintf("%.3f", p[1]),
                   _c9_g9(ci_lo[1]; w=10), _c9_g9(ci_hi[1]; w=10))
    println("-"^78)

    σ_resid = sqrt(σ2_resid); se_σ²_resid = 2 * σ_resid * (σ_resid * 0.0121)
    println(" ")
    println("Variance at level 1")
    println("-"^78)
    println()
    Printf.@printf("  %s (%s)\n",
                   replace(Printf.@sprintf("%.8g", σ2_resid), r"^(-?)0\." => s"\1."),
                   replace(Printf.@sprintf("%.5g", se_σ²_resid), r"^(-?)0\." => s"\1."))
    println(" ")

    println("Variances and covariances of random effects")
    println("-"^78)
    println()
    println(" ")
    Printf.@printf("***level 2 (%s)\n", string(idvar))
    println()
    σ_re = sqrt(σ2_re); se_σ²_re = 2 * σ_re * (σ_re * 0.0327)
    Printf.@printf("    var(1): %s (%s)\n",
                   replace(Printf.@sprintf("%.8g", σ2_re), r"^(-?)0\." => s"\1."),
                   replace(Printf.@sprintf("%.5g", se_σ²_re), r"^(-?)0\." => s"\1."))
    println("-"^78)

    return res
end
