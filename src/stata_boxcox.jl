"""
    stata_boxcox(df, depvar, xvars; condition=nothing)

Stata-style `boxcox <depvar> <xvars>` — MLE of the transformation

    y(θ) = (yᶿ − 1) / θ           (θ ≠ 0)
    y(θ) = log(y)                  (θ = 0)

Only the LHS is transformed (Stata's default `lhsonly` behaviour). Requires
`y > 0`. Use `condition = r -> r.totexp > 0` to subset.

Uses a Brent-refined golden-section search over θ ∈ [−3, 3] — no external
optimizer needed. Prints the Stata-style output and returns

    (theta, se_th, beta, sigma, loglik, nobs, xnames, ll_restricted)
"""
function stata_boxcox(df::AbstractDataFrame, depvar::Symbol,
                      xvars::Vector{Symbol}; condition = nothing)
    dfb = DataFrames.dropmissing(df[:, vcat([depvar], xvars)])
    condition !== nothing && (dfb = filter(condition, dfb))
    y = Float64.([_sm_rawval(v) for v in dfb[!, depvar]])
    X = hcat(ones(DataFrames.nrow(dfb)),
             [Float64.([_sm_rawval(v) for v in dfb[!, v]]) for v in xvars]...)
    cn = vcat(string.(xvars), "_cons")
    n, k = size(X)
    sumlogy = sum(log.(y))

    bc(y, θ) = abs(θ) < 1e-8 ? log.(y) : (y .^ θ .- 1.0) ./ θ

    function loglik(θ)
        yt   = bc(y, θ)
        β    = X \ yt
        rss  = sum((yt .- X * β) .^ 2)
        sig2 = rss / n
        return -n / 2 * log(2π * sig2) - n / 2 + (θ - 1) * sumlogy
    end

    # Golden-section maximisation of loglik over [a,b].
    function _gs_max(f, a, b; tol = 1e-6, itmax = 200)
        φ  = (sqrt(5.0) - 1) / 2
        c  = b - φ * (b - a); d = a + φ * (b - a)
        fc = f(c); fd = f(d)
        for _ in 1:itmax
            (b - a) < tol && break
            if fc > fd
                b, d, fd = d, c, fc
                c  = b - φ * (b - a); fc = f(c)
            else
                a, c, fc = c, d, fd
                d  = a + φ * (b - a); fd = f(d)
            end
        end
        return (a + b) / 2
    end

    th = _gs_max(loglik, -3.0, 3.0)

    yt   = bc(y, th)
    bhat = X \ yt
    rss  = sum((yt .- X * bhat) .^ 2)
    sig  = sqrt(rss / n)
    llhat = loglik(th)

    h    = 1e-5
    d2   = (-loglik(th + h) + 2 * loglik(th) - loglik(th - h)) / h^2
    se_th = sqrt(max(1.0 / d2, 0.0))
    z_th  = th / se_th
    p_th  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_th)))
    ci_lo = th - 1.96 * se_th
    ci_hi = th + 1.96 * se_th

    ll_restricted = Dict{Float64, Float64}()
    for t0 in (-1.0, 0.0, 1.0)
        ll_restricted[t0] = loglik(t0)
    end

    nobs_str = replace(string(n), r"(\d)(?=(\d{3})+$)" => s"\1,")
    println("\nFitting full model\n")
    Printf.@printf("%54sNumber of obs   = %10s\n", "", nobs_str)
    Printf.@printf("%54sLR chi2(%d)      = %10.2f\n",
                   "", 1, -2 * (loglik(0.0) - llhat))
    Printf.@printf("Log likelihood = %.3f%23sProb > chi2     = %10.3f\n",
                   llhat, "",
                   1 - Distributions.cdf(Distributions.Chisq(1),
                                         -2 * (loglik(0.0) - llhat)))
    println()
    println("-"^78)
    Printf.@printf("%11s | Coefficient  Std. err.      z    P>|z|     [95%% conf. interval]\n",
                   string(depvar))
    println("-"^13, "+", "-"^64)
    Printf.@printf("      /theta | %10.7f  %10.7f  %6.2f   %5.3f    %10.7f  %10.7f\n",
                   th, se_th, z_th, p_th, ci_lo, ci_hi)
    println("-"^78)
    println()
    println("Estimates of scale-variant parameters")
    println("-"^28)
    Printf.@printf("             | Coefficient\n")
    println("-"^13, "+-", "-"^13)
    println("Notrans      |")
    for (j, v) in enumerate(xvars)
        Printf.@printf("%12s | %10.7f\n", v, bhat[j + 1])
    end
    Printf.@printf("%12s | %10.6f\n", "_cons", bhat[1])
    println("-"^13, "+-", "-"^13)
    Printf.@printf("      /sigma | %10.6f\n", sig)
    println("-"^28)
    println()
    println("-"^57)
    Printf.@printf("   Test         Restricted     LR statistic\n")
    Printf.@printf("    H0:       log likelihood       chi2       Prob > chi2\n")
    println("-"^57)
    for t0 in (-1.0, 0.0, 1.0)
        ll0 = ll_restricted[t0]
        lr  = 2 * (llhat - ll0)
        p   = 1 - Distributions.cdf(Distributions.Chisq(1), lr)
        Printf.@printf("theta = %2d   %14.3f   %10.2f         %7.3f\n",
                       Int(t0), ll0, lr, p)
    end
    println("-"^57)

    return (theta = th, se_th = se_th, beta = bhat, sigma = sig,
            loglik = llhat, nobs = n, xnames = cn,
            ll_restricted = ll_restricted)
end
