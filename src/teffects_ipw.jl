# (deps provided by the StatEcon module)

# Reproduces Stata's `teffects ipw (y) (treat covars), [ate]`.
#
#   teffects_ipw(df, :train,
#                [:age, :agesq, :educ, :educsq, :age_educ, :unem96,
#                 :earn96, :earn96sq, :age_earn96, :educ_earn96];
#                outcome = :earn98)
#
# Inverse-probability-weighted estimator of the ATE:
#   1. Logit treatment model treat ~ covars  (teffects default link).
#   2. Propensity score p = Pr(treat=1 | x).
#   3. Normalised (Hajek) potential-outcome means
#        mu1 = sum(t*y/p)      / sum(t/p)
#        mu0 = sum((1-t)*y/(1-p)) / sum((1-t)/(1-p))
#      ATE = mu1 - mu0.
#
# Standard errors are the robust (sandwich) SEs from the stacked estimating
# equations, so they account for the first-step estimation of the propensity
# score -- this is what Stata's teffects reports as "Robust std. err.".

# Normal CDF via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) so no extra
# dependency (Distributions/SpecialFunctions) is needed.
function _normcdf(z::Real)
    s = sign(z); x = abs(z) / sqrt(2)
    t = 1 / (1 + 0.3275911x)
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t
             - 0.284496736)t + 0.254829592)t * exp(-x^2)
    return 0.5 * (1 + s * y)
end
_pval(z) = 2 * (1 - _normcdf(abs(z)))

function teffects_ipw(df::AbstractDataFrame, treat::Symbol, covars::AbstractVector{Symbol};
                      outcome::Symbol)

    keep = completecases(df[:, unique([treat; outcome; covars])])   # e(sample)
    work = df[keep, :]
    y = Float64.(work[!, outcome])
    t = Float64.(work[!, treat])

    # ---- 1. logit treatment model & propensity score ------------------
    fm = Term(treat) ~ sum(Term.(covars))
    m  = GLM.glm(fm, work, Binomial(), LogitLink())
    X  = GLM.modelmatrix(m)                    # N x K, intercept in column 1
    p  = GLM.predict(m)
    N, K = size(X)

    # ---- 2. normalised IPW potential-outcome means --------------------
    mu1 = sum(t .* y ./ p)          / sum(t ./ p)
    mu0 = sum((1 .- t) .* y ./ (1 .- p)) / sum((1 .- t) ./ (1 .- p))
    ATE = mu1 - mu0

    # ---- 3. stacked-equations sandwich variance -----------------------
    # per-observation scores s_i = [ (t-p)x ; t(y-mu1)/p ; (1-t)(y-mu0)/(1-p) ]
    S = zeros(N, K + 2)
    S[:, 1:K] .= (t .- p) .* X
    S[:, K+1] .= t .* (y .- mu1) ./ p
    S[:, K+2] .= (1 .- t) .* (y .- mu0) ./ (1 .- p)

    # expected Jacobian A = (1/N) sum d s_i / d theta'
    A = zeros(K + 2, K + 2)
    A[1:K, 1:K] .= -(X' * (p .* (1 .- p) .* X)) / N
    A[K+1, 1:K] .= -(X' * (t .* (y .- mu1) .* (1 .- p) ./ p)) / N
    A[K+1, K+1]  = -sum(t ./ p) / N
    A[K+2, 1:K] .=  (X' * ((1 .- t) .* (y .- mu0) .* p ./ (1 .- p))) / N
    A[K+2, K+2]  = -sum((1 .- t) ./ (1 .- p)) / N

    B  = (S' * S) / N
    Ai = LinearAlgebra.inv(A)
    V  = (Ai * B * Ai') / N                    # var(theta)

    L        = [zeros(K); 1.0; -1.0]           # ATE = mu1 - mu0
    se_ate   = sqrt(L' * V * L)
    se_mu0   = sqrt(V[K+2, K+2])
    se_mu1   = sqrt(V[K+1, K+1])

    # ---- teffects-style output ----------------------------------------
    function row(label, est, se)
        z  = est / se
        lo = est - 1.959964se; hi = est + 1.959964se
        Printf.@printf("%-14s | %10.5g %10.5g %7.2f %8.3f %11.5g %11.5g\n",
                label, est, se, z, _pval(z), lo, hi)
    end
    Printf.@printf("%-14s | %10s %10s %7s %8s %11s %11s\n",
            string(outcome), "Coef.", "Robust SE", "z", "P>|z|", "[95% conf.", "interval]")
    println("-"^14 * "+" * "-"^61)
    println("ATE")
    row("  (1 vs 0)", ATE, se_ate)
    println("POmean")
    row("  control (0)", mu0, se_mu0)

    return DataFrames.DataFrame(Parameter = ["ATE", "POmean_0", "POmean_1"],
                     Estimate  = [ATE, mu0, mu1],
                     RobustSE  = [se_ate, se_mu0, se_mu1])
end
