# ============================================================================
# stata_estat_abond.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_estat_abond(res; orders=1:2)

Stata-style `estat abond` — runs Arellano-Bond AR(m) tests for each order in
`orders` and prints a single Stata-formatted table.
"""
function stata_estat_abond(res; orders=1:2)
    println("\nArellano-Bond test for zero autocorrelation in first-differenced errors")
    println("H0: No autocorrelation \n")
    println("Order         z   Prob > z")
    println("-"^26)
    results = NamedTuple[]
    for m in orders
        r = stata_estat_abond_artest(res, m)
        zstr = Printf.@sprintf("%.4g", r.z)
        if 0 < abs(r.z) < 1
            zstr = replace(zstr, r"^(-?)0\." => s"\1.")
        end
        Printf.@printf("%5d   %7s     %.4f\n", m, zstr, r.p)
        push!(results, r)
    end
    println("-"^26)
    return results
end
