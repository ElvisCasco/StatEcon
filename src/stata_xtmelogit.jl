# (deps provided by the StatEcon module)

"""
    stata_xtmelogit(df; depvar, regs, idvar, integration_pts=7,
                    level=0.95, quiet=false) -> NamedTuple

Stata-style `xtmelogit <depvar> <regs> || <id>:` (also `meqrlogit`,
`melogit`). Same random-intercept logit model as `xtlogit, re` (the
underlying estimator is identical), but the printed output uses Stata's
`Mixed-effects logistic regression` layout with the dedicated
`Random-effects parameters` block. Stata's default integration points
for `xtmelogit` is 7 (vs 12 for `xtlogit, re`).

Internally calls `stata_xtlogit_re(...; quiet=true)` and reformats the
output. All numerical results are identical to `stata_xtlogit_re` for
the same `integration_pts`.

Returns the same NamedTuple as `stata_xtlogit_re`.
"""
function stata_xtmelogit(df; depvar::Symbol,
                         regs::AbstractVector{Symbol},
                         idvar::Symbol,
                         integration_pts::Int = 7,
                         level::Float64 = 0.95,
                         quiet::Bool = false)
    r = stata_xtlogit_re(df; depvar = depvar, regs = regs, idvar = idvar,
                         integration_pts = integration_pts,
                         level = level, quiet = true)

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w)
    end
    commafmt(num) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    if !quiet
        crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
        n     = r.n
        n_pan = r.n_panels
        β     = r.β
        se    = r.se
        cnames = r.coefnames
        k     = length(β)

        z     = β ./ se
        pv    = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
        ci_lo = β .- crit .* se
        ci_hi = β .+ crit .* se

        # Header.
        println()
        Printf.@printf("%-48s%-18s= %10s\n",
                       "Mixed-effects logistic regression",
                       "Number of obs", commafmt(n))
        Printf.@printf("%-48s%-18s= %10s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", string(n_pan))
        println()
        Printf.@printf("%48s%s\n", "", "Obs per group:")
        Printf.@printf("%48s%18s = %10d\n", "", "min", r.T_min)
        Printf.@printf("%48s%18s = %10.1f\n", "", "avg", r.T_avg)
        Printf.@printf("%48s%18s = %10d\n", "", "max", r.T_max)
        println()
        Printf.@printf("%-48s%-18s= %10s\n",
                       "Integration points = " *
                           lpad(string(integration_pts), 3),
                       "Wald chi2($(k - 1))",
                       Printf.@sprintf("%10.2f", r.Wald))
        ll_str = Printf.@sprintf("Log likelihood = %.3f", r.ll)
        right  = Printf.@sprintf("%-18s= %10s", "Prob > chi2",
                                 Printf.@sprintf("%.4f", r.Wald_p))
        pad_h  = max(0, 78 - length(ll_str) - length(right))
        println(ll_str, " "^pad_h, right)
        println()

        # Fixed-effects coefficient table.
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100*level)
        println("-"^13, "+", "-"^64)
        for i in 1:k
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           cnames[i], g9(β[i]; w=10), g9(se[i]; w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", pv[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^78)

        # Random-effects parameters block: only sd(_cons) here.
        println()
        println("-"^78)
        ci_label = Printf.@sprintf("[%d%% conf. interval]", round(Int, 100*level))
        println("  Random-effects parameters  |   Estimate   Std. err.     ", ci_label)
        println("-"^29, "+", "-"^48)
        Printf.@printf("%-29s|\n", string(idvar) * ": Identity")
        Printf.@printf("%29s| %10s  %10s  %10s  %10s\n",
                       lpad("sd(_cons)", 28),
                       g9(r.σ_u; w=10), g9(r.se_σu; w=10),
                       g9(r.σu_lo; w=10), g9(r.σu_hi; w=10))
        println("-"^78)
        Printf.@printf("LR test vs. logistic model: chibar2(01) = %.2f%5sProb >= chibar2 = %.4f\n",
                       r.LR, "", r.p_chibar)
    end

    return r
end

