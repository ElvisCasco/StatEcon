import Optim
import ForwardDiff

# --------------------------------------------------------------------------
# stata_streg — parametric survival (duration) regression in Stata's
# proportional-hazards (log relative-hazard) parameterisation. Reproduces
#   streg <covariates>, dist(weibull|exponential) [nohr]
# The log likelihood follows Stata's streg convention (the d·ln t Jacobian is
# dropped) so `ll` is directly comparable with the published output.
# (deps: DataFrames, LinearAlgebra, Printf, Distributions, Optim, ForwardDiff)
# --------------------------------------------------------------------------

# streg fit block (nohr = coefficients, otherwise hazard ratios). β holds the
# regression coefficients (_cons last); for weibull the shape enters as ln p.
function _streg_print(dist::Symbol, nohr::Bool, level::Float64,
                      n::Int, d, t, k::Int, LR::Float64, ll::Float64,
                      θ̂, se, cnames)
    weib = dist === :weibull
    zc = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    dname = weib ? "Weibull" : "Exponential"
    println("$dname regression -- log relative-hazard form\n")
    Printf.@printf("No. of subjects = %12d                     Number of obs   = %9d\n", n, n)
    Printf.@printf("No. of failures = %12d\n", Int(sum(d)))
    tar = sum(t)
    if isinteger(tar)
        Printf.@printf("Time at risk    = %12d\n", Int(tar))
    else
        Printf.@printf("Time at risk    = %12.3f\n", tar)
    end
    Printf.@printf("%51s LR chi2(%d)     = %9.2f\n", "", k, LR)
    Printf.@printf("Log likelihood  = %12.4f                     Prob > chi2     = %9.4f\n",
                   ll, 1 - Distributions.cdf(Distributions.Chisq(k), LR))
    println()
    println("-"^78)
    collab = nohr ? "Coef." : "Haz. Ratio"
    Printf.@printf("%12s | %10s %10s %7s %6s  %10s %10s\n",
                   "_t", collab, "Std. Err.", "z", "P>|z|", "[95% Conf.", "Interval]")
    println("-"^13 * "+" * "-"^64)
    nb = k + 1                                   # covariates + _cons
    for i in 1:nb
        z = θ̂[i] / se[i]
        pv = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        if nohr
            Printf.@printf("%12s | %10.7f %10.7f %7.2f %6.3f  %10.7f %10.7f\n",
                           cnames[i], θ̂[i], se[i], z, pv,
                           θ̂[i] - zc * se[i], θ̂[i] + zc * se[i])
        else
            hr = exp(θ̂[i])
            Printf.@printf("%12s | %10.7f %10.7f %7.2f %6.3f  %10.7f %10.7f\n",
                           cnames[i], hr, hr * se[i], z, pv,
                           exp(θ̂[i] - zc * se[i]), exp(θ̂[i] + zc * se[i]))
        end
    end
    if weib
        println("-"^13 * "+" * "-"^64)
        lnp = θ̂[end]; s = se[end]; z = lnp / s
        Printf.@printf("%12s | %10.7f %10.7f %7.2f %6.3f  %10.7f %10.7f\n",
                       "/ln_p", lnp, s, z,
                       2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z))),
                       lnp - zc * s, lnp + zc * s)
        println("-"^13 * "+" * "-"^64)
        p = exp(lnp)
        Printf.@printf("%12s | %10.7f %10.7f %7s %6s  %10.7f %10.7f\n",
                       "p", p, p * s, "", "", exp(lnp - zc * s), exp(lnp + zc * s))
        Printf.@printf("%12s | %10.7f %10.7f %7s %6s  %10.7f %10.7f\n",
                       "1/p", 1 / p, s / p, "", "",
                       exp(-(lnp + zc * s)), exp(-(lnp - zc * s)))
    end
    println("-"^78)
end

"""
    stata_streg(df, time, failure, covariates;
                dist=:weibull, nohr=false, level=0.95, quiet=false) -> NamedTuple

Parametric survival (duration) regression — Stata
`streg <covariates>, dist(<dist>) [nohr]`. Fits the proportional-hazards
(log relative-hazard) parameterisation by maximum likelihood with Stata's
convention (the `d·ln t` Jacobian is dropped, so the log likelihood matches
`streg`).

* `time`       — analysis-time variable (Stata's `_t`).
* `failure`    — failure indicator (1 = event, 0 = right-censored).
* `covariates` — regressors; an intercept `_cons` is always added.
* `dist`       — `:weibull` (shape `p`, reported as `/ln_p`, `p`, `1/p`) or
                 `:exponential` (`p ≡ 1`, no ancillary rows).
* `nohr`       — `true` prints coefficients (`Coef.`, log relative-hazard,
                 verified against Stata's `streg … nohr`); `false` (Stata's
                 default) prints hazard ratios `exp(β)`.

Prints the `streg` header (subjects / failures / time at risk / LR χ²(k) / log
likelihood) and the coefficient table, and returns
`(; β, se, V, ll, ll0, LR, lnp, p, coefnames, n, nfail, dist)`. `β` holds the
regression coefficients (`_cons` last), followed by `ln p` when `dist == :weibull`.
"""
function stata_streg(df, time, failure, covariates;
                     dist::Symbol = :weibull, nohr::Bool = false,
                     level::Float64 = 0.95, quiet::Bool = false)
    (dist === :weibull || dist === :exponential) ||
        error("dist must be :weibull or :exponential")
    xs   = [Symbol(v) for v in covariates]
    tsym = Symbol(time); dsym = Symbol(failure)
    dfc  = DataFrames.dropmissing(df[:, unique(vcat(tsym, dsym, xs))])
    n    = DataFrames.nrow(dfc)
    k    = length(xs)
    X    = hcat(Matrix{Float64}(dfc[:, xs]), ones(n))    # intercept last
    t    = Float64.(dfc[!, tsym])
    d    = Float64.(dfc[!, dsym])
    logt = log.(t)
    weib = dist === :weibull

    # streg (PH) log likelihood, ln t Jacobian dropped. p ≡ 1 for exponential.
    nll = function (θ)
        if weib
            β = @view θ[1:end-1]; lp = θ[end]; p = exp(lp)
        else
            β = θ; lp = zero(eltype(θ)); p = one(eltype(θ))
        end
        xb = X * β; s = zero(eltype(θ))
        @inbounds for i in eachindex(t)
            s += d[i] * (lp + p * logt[i] + xb[i]) - t[i]^p * exp(xb[i])
        end
        return -s
    end
    θ0 = zeros(size(X, 2) + (weib ? 1 : 0))
    θ0[size(X, 2)] = -3.0                                 # intercept start
    g! = (g, x) -> ForwardDiff.gradient!(g, nll, x)
    opt = Optim.optimize(nll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-12, iterations = 5_000))
    θ̂  = Optim.minimizer(opt)
    V   = LinearAlgebra.inv(ForwardDiff.hessian(nll, θ̂))
    se  = sqrt.(LinearAlgebra.diag(V))
    ll  = -Optim.minimum(opt)

    # constant-only model → Stata's LR χ²(k)
    nll0 = function (θ)
        b = θ[1]
        if weib
            lp = θ[2]; p = exp(lp)
        else
            lp = zero(eltype(θ)); p = one(eltype(θ))
        end
        s = zero(eltype(θ))
        @inbounds for i in eachindex(t)
            s += d[i] * (lp + p * logt[i] + b) - t[i]^p * exp(b)
        end
        return -s
    end
    θ0c  = weib ? [-3.0, 0.0] : [-3.0]
    g0!  = (g, x) -> ForwardDiff.gradient!(g, nll0, x)
    opt0 = Optim.optimize(nll0, g0!, θ0c, Optim.LBFGS(),
                          Optim.Options(g_tol = 1e-12, iterations = 5_000))
    ll0  = -Optim.minimum(opt0)
    LR   = 2 * (ll - ll0)

    cnames = vcat(string.(xs), "_cons")
    lnp    = weib ? θ̂[end] : 0.0

    quiet || _streg_print(dist, nohr, level, n, d, t, k, LR, ll, θ̂, se, cnames)
    return (; β = θ̂, se, V, ll, ll0, LR, lnp, p = exp(lnp),
              coefnames = cnames, n = n, nfail = Int(sum(d)), dist = dist)
end
