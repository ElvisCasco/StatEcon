# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_estat_gof — `estat gof, group(g)` Hosmer–Lemeshow GOF test
# --------------------------------------------------------------------------

"""
    stata_estat_gof(y, p, g; depvar="", model_label="logistic", quiet=false)
        -> NamedTuple

Stata's `estat gof, group(g)` Hosmer–Lemeshow goodness-of-fit test.
Sorts by predicted probability `p`, partitions into `g` ~equal-sized
groups, then computes

    χ² = Σ_i [(O_{1i} − E_{1i})² / E_{1i} + (O_{0i} − E_{0i})² / E_{0i}]

under H0: model is correctly specified, ~ χ²(g − 2).

When `depvar` is provided, prints Stata's full block:

    Goodness-of-fit test after <model_label> model
    Variable: <depvar>

     Number of observations = …
           Number of groups = …
    Hosmer–Lemeshow chi2(g-2) = …
                Prob > chi2 = …

Returns `(; chi2, df, pvalue, n, g)`.
"""
function stata_estat_gof(y::AbstractVector, p::AbstractVector, g::Int;
                         depvar::AbstractString = "",
                         model_label::AbstractString = "logistic",
                         quiet::Bool = false)
    n   = length(y)
    idx = sortperm(p)
    y_s = y[idx]; p_s = p[idx]
    grp = div(n, g)
    χ2  = 0.0
    for i in 1:g
        lo = (i - 1) * grp + 1
        hi = i == g ? n : i * grp
        ng = hi - lo + 1
        o1 = sum(y_s[lo:hi]); e1 = sum(p_s[lo:hi])
        o0 = ng - o1;          e0 = ng - e1
        e1 > 0 && (χ2 += (o1 - e1)^2 / e1)
        e0 > 0 && (χ2 += (o0 - e0)^2 / e0)
    end
    df = g - 2
    pv = 1 - Distributions.cdf(Distributions.Chisq(df), χ2)
    if !quiet
        commafmt(num) = begin
            s = string(abs(num)); parts = String[]; i = length(s)
            while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
            (num < 0 ? "-" : "") * join(reverse(parts), ",")
        end
        if !isempty(depvar)
            println("Goodness-of-fit test after $model_label model")
            println("Variable: $depvar")
            println()
        end
        Printf.@printf("%23s = %7s\n",  "Number of observations", commafmt(n))
        Printf.@printf("%23s = %7d\n",  "Number of groups", g)
        Printf.@printf("%23s = %7.2f\n", "Hosmer–Lemeshow chi2($df)", χ2)
        Printf.@printf("%23s = %7.4f\n", "Prob > chi2", pv)
    end
    return (; chi2 = χ2, df, pvalue = pv, n, g)
end
