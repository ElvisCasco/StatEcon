# ============================================================================
# stata_nlcom.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_nlcom(f, β, V; name="_nl_1", depvar="", level=0.95)

Stata-style `nlcom <expression>` — point estimate of an arbitrary smooth
function `f(β)` of the coefficient vector, with delta-method SE using vcov V:

    SE(f(β̂)) = sqrt( ∇f(β̂)' · V · ∇f(β̂) )

Reports a Stata-style table with z = f(β̂)/SE, two-sided p, and a normal CI.
The gradient is computed by central finite differences. Pass β and V from the
relevant model (e.g. the n/(n−1)-corrected robust V from `stata_poisson`).
"""
function stata_nlcom(f::Function, β::AbstractVector, V::AbstractMatrix;
                     name::String="_nl_1", depvar::String="",
                     level::Float64=0.95)
    val = f(β)
    n_p = length(β)
    g   = zeros(n_p)
    for j in 1:n_p
        h    = sqrt(eps(Float64)) * max(abs(β[j]), 1.0)
        βp   = collect(β); βm = collect(β)
        βp[j] += h; βm[j] -= h
        g[j]  = (f(βp) - f(βm)) / (2h)
    end
    se   = sqrt(g' * V * g)
    z    = val / se
    p    = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    lo   = val - crit * se
    hi   = val + crit * se

    g9 = function(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return lpad(s, w)
    end

    println("-"^78)
    Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                   depvar, 100*level)
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   name, g9(val; w=10), g9(se; w=9),
                   Printf.@sprintf("%7.2f", z),
                   Printf.@sprintf("%.3f", p),
                   g9(lo; w=9), g9(hi; w=10))
    println("-"^78)
    return (; val, se, z, p, ci=(lo, hi))
end
