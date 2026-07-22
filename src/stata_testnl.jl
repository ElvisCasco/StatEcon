# ============================================================================
# stata_testnl.jl — Stata `testnl` (Cameron & Trivedi ch12: Testing methods)
# ============================================================================

import ForwardDiff

"""
    stata_testnl(g, β, V; label="g(β) = 0") -> (; chi2, df, p)

Stata-style `testnl <expression>` — delta-method Wald χ² for a nonlinear
hypothesis `g(β) = 0`. Linearises `g` around `β̂` via ForwardDiff and returns

    χ² = g(β̂)' · (∇g' V ∇g)⁻¹ · g(β̂),   df = length(g(β̂)).

`g` is a closure mapping `β → scalar` (df = 1) — e.g. `β -> β[i_f]/β[i_p] - 1`
for `_b[female]/_b[private] = 1` — or `β → vector` for a joint nonlinear test
(df = q). `label` is the Stata-style hypothesis string printed above the χ²
block.

Pass `V` from a robust vcov (e.g. `setup_mus10().V_robust`) to match Stata's
`testnl` after `vce(robust)`. Returns `(; chi2, df, p)`.
"""
function stata_testnl(g::Function, β::AbstractVector, V::AbstractMatrix;
                      label::AbstractString="g(β) = 0")
    x    = collect(float.(β))
    gval = g(x)
    if gval isa Real
        ∇g  = ForwardDiff.gradient(g, x)
        var = (∇g' * V * ∇g)::Real
        W   = gval^2 / var
        df  = 1
    else
        gv  = collect(float.(gval))
        J   = ForwardDiff.jacobian(g, x)
        W   = (gv' * (LinearAlgebra.Symmetric(J * V * J') \ gv))::Real
        df  = length(gv)
    end
    p = Distributions.ccdf(Distributions.Chisq(df), W)
    println()
    Printf.@printf("  (1)  %s\n\n", label)
    Printf.@printf("               chi2(%d) = %11.2f\n", df, W)
    Printf.@printf("           Prob > chi2 = %13.4f\n", p)
    return (; chi2 = W, df, p)
end
