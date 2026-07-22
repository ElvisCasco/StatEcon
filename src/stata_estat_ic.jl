# ============================================================================
# stata_estat_ic.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_estat_ic(fit; model_name=".")

Stata-style `estat ic` — Akaike and Bayesian information criteria. `fit` is a
NamedTuple with fields `n`, `k`, `ll`, `pseudo_r2` (e.g. a `stata_poisson`
return value).

  - `ll(null)` is recovered from `ll / (1 − pseudo_r2)`.
  - `df` reported is `k` (number of estimated parameters).
  - AIC = −2·ll + 2·k,   BIC = −2·ll + k·log(n).

Returns `(; n, k, ll, ll_null, aic, bic)`.
"""
function stata_estat_ic(fit; model_name::String=".")
    n  = fit.n
    k  = fit.k
    ll = fit.ll
    ll_null = ll / (1 - fit.pseudo_r2)
    aic = -2*ll + 2*k
    bic = -2*ll + k*log(n)

    commafmt(num::Integer) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    println("\nAkaike's information criterion and Bayesian information criterion\n")
    println("-"^78)
    println("       Model |          N   ll(null)  ll(model)      df        AIC        BIC")
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %10s %10.6g %10.7g %7d %10.6g %10.7g\n",
                   model_name, commafmt(n), ll_null, ll, k, aic, bic)
    println("-"^78)
    return (; n, k, ll, ll_null, aic, bic)
end
