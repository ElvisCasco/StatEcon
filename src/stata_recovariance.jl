# ============================================================================
# stata_recovariance.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_recovariance(res; print_table=true)

Stata's `estat recovariance` — prints the (lower-triangular) covariance matrix
of the random effects. `res` is a NamedTuple returned by `stata_xtmixed`
(fields `Ψ` and `re_names`). Returns the covariance matrix.
"""
function stata_recovariance(res; print_table::Bool = true)
    Σ = res.Ψ
    names = hasproperty(res, :re_names) ? res.re_names : ["x$(i)" for i in 1:size(Σ, 1)]
    q = size(Σ, 1)
    if print_table
        disp = [nm == "_cons" ? "_cons" : nm for nm in names]
        println("\nRandom-effects covariance matrix for level $(res.idvar)\n")
        Printf.@printf("%12s |", "")
        for nm in disp;  Printf.@printf(" %9s ", nm);  end
        println()
        println("-"^13, "+", "-"^(11 * q + 1))
        for i in 1:q
            Printf.@printf("%12s |", disp[i])
            for j in 1:q
                if j <= i
                    Printf.@printf(" %s ", _c9_g9(Σ[i, j]; w=9))
                else
                    Printf.@printf(" %9s ", "")
                end
            end
            println()
        end
    end
    return Σ
end
