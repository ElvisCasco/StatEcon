# ============================================================================
# stata_lincom.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_lincom(w, β, V; eform=false, c=0.0, name="(1)", depvar="", level=0.95)

Stata-style `lincom <linear combination>` (optionally `, eform`). Computes
`L = w'β + c`, its delta-method SE = sqrt(w' V w), z = L/SE(L), two-sided p,
and a CI on the linear scale. With `eform=true`, reports `exp(L)` and
`exp(L)·SE(L)` and CI = `exp(L ± z·SE(L))` (asymmetric) — but **z and p stay
on the LINEAR scale** (the key difference from `nlcom exp(w'β)`).

`c` is an additive constant in the combination (e.g. `... - 1` → `c = -1`);
it shifts L but not the SE. Pass β and V from the relevant model.
"""
function stata_lincom(w::AbstractVector, β::AbstractVector, V::AbstractMatrix;
                      eform::Bool=false, c::Float64=0.0, name::String="(1)",
                      depvar::String="", level::Float64=0.95)
    L    = LinearAlgebra.dot(w, β) + c
    seL  = sqrt(w' * V * w)
    z    = L / seL
    p    = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)

    if eform
        val, se = exp(L), exp(L) * seL
        lo, hi  = exp(L - crit * seL), exp(L + crit * seL)
        clab    = "exp(b)"
    else
        val, se = L, seL
        lo, hi  = L - crit * seL, L + crit * seL
        clab    = "Coefficient"
    end

    g9 = function(x; w_::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w_)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w_)
    end

    label_field = rpad(lpad(clab, 10), 13)
    println("-"^78)
    Printf.@printf("%12s | %sStd. err.      z    P>|z|     [%g%% conf. interval]\n",
                   depvar, label_field, 100*level)
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   name, g9(val; w_=10), g9(se; w_=9),
                   Printf.@sprintf("%7.2f", z),
                   Printf.@sprintf("%.3f", p),
                   g9(lo; w_=9), g9(hi; w_=10))
    println("-"^78)
    return (; val, se, z, p, ci=(lo, hi))
end
