# ---------------------------------------------------------------------------
# Reproducible post-estimation and time-series helpers used across the worked
# examples. Each mirrors a Stata post-estimation tool that has no single
# dedicated StatEcon command:
#
#   lincom          -> `lincom <combination>`      (t-based linear combination)
#   stdbeta         -> `regress ..., beta`          (standardized coefficients)
#   predict_ci      -> `predict, stdp` / `, stdf`   (prediction SE / forecast SE)
#   sigma2_of       -> residual variance (Stata's Residual MS) of a fitted model
#   newey_west      -> `newey y x, lag(L)`          (HAC SEs, Bartlett kernel)
#   cochrane_orcutt -> `prais y x, corc`            (iterated AR(1))
#   bpagan_lm       -> `bpagan x`                    (Breusch-Pagan 1979 LM stat)
#   sargan          -> Sargan overidentification test (multi-endogenous 2SLS)
#   ivreg2_table    -> rescale a 2SLS fit to `ivreg2`'s large-sample SEs
#
# (deps provided by the StatEcon module: DataFrames, Statistics, StatsBase,
#  LinearAlgebra, Printf, Distributions, FixedEffectModels)
# ---------------------------------------------------------------------------

# internal: coefficient names as plain strings, including "(Intercept)"
_eh_cn(m) = string.(StatsBase.coefnames(m))
# internal: design matrix [1 X] in the current (assumed time-sorted) row order
_eh_desX(df, xs) = hcat(ones(DataFrames.nrow(df)),
                        (Float64.(df[!, c]) for c in xs)...)
# internal: Durbin-Watson statistic from a time-ordered residual vector
_eh_dw(e) = sum((e[2:end] .- e[1:end-1]).^2) / sum(e.^2)

"""
    lincom(m, terms; c=0.0, level=0.95) -> NamedTuple

Stata `lincom` — a linear combination `w'β + c` of the coefficients of a fitted
model `m`, with a `t`-based standard error, statistic, p-value and confidence
interval on the model's residual degrees of freedom. `terms` is an iterable of
`name => weight` pairs where `name` is a coefficient name exactly as the model
prints it (use `"(Intercept)"` for the constant). Prints the Stata-style line
and returns `(; est, se, t, p, lo, hi)`.
"""
function lincom(m, terms; c::Real = 0.0, level::Real = 0.95)
    names = _eh_cn(m)
    b = StatsBase.coef(m); V = StatsBase.vcov(m)
    w = zeros(length(b))
    for (nm, wt) in terms
        i = findfirst(==(string(nm)), names)
        i === nothing && error("no coefficient named $nm")
        w[i] += wt
    end
    est = LinearAlgebra.dot(w, b) + c
    se  = sqrt(LinearAlgebra.dot(w, V * w))
    dfr = StatsBase.dof_residual(m)
    t   = est / se
    p   = 2 * (1 - Distributions.cdf(Distributions.TDist(dfr), abs(t)))
    tc  = Distributions.quantile(Distributions.TDist(dfr), 1 - (1 - level) / 2)
    Printf.@printf("  estimate = %.7g   se = %.7g\n  t = %.2f   p = %.3f   %g%% CI = [%.7g, %.7g]\n",
                   est, se, t, p, 100 * level, est - tc * se, est + tc * se)
    return (; est, se, t, p, lo = est - tc * se, hi = est + tc * se)
end

"""
    stdbeta(df, y, xs) -> model

Stata `regress ..., beta`: fits `stata_regress(df; y, x=xs)` and prints each
slope's standardized coefficient `bⱼ · sd(xⱼ)/sd(y)` over the estimation sample.
Returns the fitted model.
"""
function stdbeta(df, y, xs)
    sy = Statistics.std(df[!, y])
    m  = stata_regress(df; y = y, x = xs)
    names = _eh_cn(m); b = StatsBase.coef(m)
    for x in xs
        bj = b[findfirst(==(string(x)), names)]
        Printf.@printf("  %-10s beta = % .7g\n", x, bj * Statistics.std(df[!, x]) / sy)
    end
    return m
end

"""
    predict_ci(m, x0; forecast=false, sigma2=nothing) -> NamedTuple

Point prediction `x0'β` and its standard error for a new regressor row `x0`
(in coefficient order, intercept first). With `forecast=false` this is Stata's
`predict, stdp` (mean-prediction SE); with `forecast=true` it adds `sigma2` to
the variance for `predict, stdf` (forecast SE). Prints the value, SE and 95% CI
and returns `(; xb, se)`.
"""
function predict_ci(m, x0; forecast::Bool = false, sigma2 = nothing)
    b = StatsBase.coef(m); V = StatsBase.vcov(m)
    xb   = LinearAlgebra.dot(b, x0)
    varp = LinearAlgebra.dot(x0, V * x0)
    forecast && (varp += sigma2)
    sd = sqrt(varp)
    Printf.@printf("  xb = %.7g   se = %.7g\n  95%% CI = [%.7g, %.7g]\n",
                   xb, sd, xb - 1.96sd, xb + 1.96sd)
    return (xb = xb, se = sd)
end

"""
    sigma2_of(m, df, y) -> Float64

Residual variance `σ̂² = RSS / dof_residual` of a fitted `FixedEffectModels`
model (Stata's Residual MS), computed from `y − predict(m, df)` over the rows
of `df`. Feed the result to [`predict_ci`](@ref) as `sigma2` for a forecast SE.
"""
function sigma2_of(m, df, y)
    r  = df[!, y] .- FixedEffectModels.predict(m, df)
    rr = collect(skipmissing(r))
    sum(abs2, rr) / StatsBase.dof_residual(m)
end

"""
    newey_west(df, y, xs, L) -> NamedTuple

Stata `newey y xs, lag(L)`: OLS of `y` on `[1 xs]` with Newey-West HAC standard
errors (Bartlett kernel, `L` lags) and Stata's `n/(n-k)` small-sample factor.
Rows are taken in their current order, assumed time-sorted. Returns
`(; b, se, t, n, k)`.
"""
function newey_west(df, y, xs, L)
    X = _eh_desX(df, xs); yy = Float64.(df[!, y])
    n, k = size(X); bread = LinearAlgebra.inv(X'X)
    b = bread * (X'yy); e = yy .- X * b
    S = zeros(k, k)
    for t in 1:n
        S .+= e[t]^2 * (X[t, :] * X[t, :]')
    end
    for l in 1:L
        w = 1 - l / (L + 1); G = zeros(k, k)
        for t in (l + 1):n
            G .+= e[t] * e[t - l] * (X[t, :] * X[t - l, :]' + X[t - l, :] * X[t, :]')
        end
        S .+= w * G
    end
    V  = (n / (n - k)) * bread * S * bread
    se = sqrt.(LinearAlgebra.diag(V))
    (b = b, se = se, t = b ./ se, n = n, k = k)
end

"""
    cochrane_orcutt(df, y, xs; tol=1e-6, maxit=200) -> NamedTuple

Iterated Cochrane-Orcutt AR(1) regression of `y` on `[1 xs]` (Stata
`prais y xs, corc`). Rows are taken in their current order, assumed time-sorted.
Returns `(; b, se, t, rho, n, k, dw_orig, dw_trans)` where the Durbin-Watson
statistics are for the original and quasi-differenced residuals.
"""
function cochrane_orcutt(df, y, xs; tol::Real = 1e-6, maxit::Int = 200)
    X = _eh_desX(df, xs); yy = Float64.(df[!, y]); n, k = size(X)
    b = (X'X) \ (X'yy); rho = 0.0
    for _ in 1:maxit
        e = yy .- X * b
        rnew = sum(e[2:end] .* e[1:end-1]) / sum(e[1:end-1].^2)
        yt = yy[2:end] .- rnew .* yy[1:end-1]
        Xt = X[2:end, :] .- rnew .* X[1:end-1, :]
        b  = (Xt'Xt) \ (Xt'yt)
        abs(rnew - rho) < tol && (rho = rnew; break)
        rho = rnew
    end
    yt = yy[2:end] .- rho .* yy[1:end-1]
    Xt = X[2:end, :] .- rho .* X[1:end-1, :]
    nt = length(yt); bt = (Xt'Xt) \ (Xt'yt); et = yt .- Xt * bt
    s2 = sum(et.^2) / (nt - k); se = sqrt.(LinearAlgebra.diag(s2 * LinearAlgebra.inv(Xt'Xt)))
    e0 = yy .- X * ((X'X) \ (X'yy))
    (b = bt, se = se, t = bt ./ se, rho = rho, n = nt, k = k,
     dw_orig = _eh_dw(e0), dw_trans = _eh_dw(et))
end

"""
    bpagan_lm(df, y, xs, zs) -> Float64

Breusch-Pagan (1979) LM statistic for heteroskedasticity of the OLS of `y` on
`[1 xs]`, testing whether the scaled squared residuals depend on `[1 zs]`
(the user-written `bpagan`). Returns the LM statistic (χ² with `length(zs)` df).
"""
function bpagan_lm(df, y, xs, zs)
    X = _eh_desX(df, xs); yy = Float64.(df[!, y]); n = size(X, 1)
    e = yy .- X * ((X'X) \ (X'yy)); g = e.^2 ./ (sum(e.^2) / n)
    Z = _eh_desX(df, zs); gh = Z * ((Z'Z) \ (Z'g))
    sum((gh .- Statistics.mean(g)).^2) / 2
end

"""
    sargan(df, y, endog, exog, ivs) -> NamedTuple

Sargan overidentification test (`n·R²` form) for a 2SLS regression of `y` on
`endog` (instrumented) and `exog` (included), using excluded instruments `ivs`.
Handles more than one endogenous regressor. Prints the Stata-style two-line
report and returns `(; chi2, df, p)` with `df = length(ivs) - length(endog)`.
"""
function sargan(df, y, endog, exog, ivs)
    n  = DataFrames.nrow(df)
    Z  = hcat(ones(n), [Float64.(df[!, v]) for v in vcat(exog, ivs)]...)
    Xe = hcat([Float64.(df[!, v]) for v in endog]...)
    X  = hcat(ones(n), Z * (Z \ Xe), [Float64.(df[!, v]) for v in exog]...)
    Xr = hcat(ones(n), [Float64.(df[!, v]) for v in vcat(endog, exog)]...)
    yv = Float64.(df[!, y])
    u  = yv .- Xr * (X \ yv)
    r  = u .- Z * (Z \ u)
    R2 = 1 - sum(abs2, r) / sum(abs2, u)
    chi2 = n * R2
    dfree = length(ivs) - length(endog)
    pv = 1 - Distributions.cdf(Distributions.Chisq(dfree), chi2)
    Printf.@printf("Sargan statistic (overidentification test of all instruments): %8.3f\n", chi2)
    Printf.@printf("                                        Chi-sq(%d) P-val = %9.5f\n", dfree, pv)
    return (; chi2, df = dfree, p = pv)
end

"""
    ivreg2_table(m; dep="y") -> NamedTuple

Rescale a fitted 2SLS model `m` (from [`stata_ivregress_2sls`](@ref), built on
`FixedEffectModels` with the `N−K` small-sample divisor) to the user-written
`ivreg2` convention with no `small` option: large-sample (`N` divisor) standard
errors and `z` statistics, `seₗₐᵣ𝓰ₑ = se·√((N−K)/N)`. Coefficients are
unchanged. Prints the table and returns `(; β, se, z, p, names)`.
"""
function ivreg2_table(m; dep::AbstractString = "y")
    β  = StatsBase.coef(m)
    n  = Int(StatsBase.nobs(m))
    k  = length(β)
    se = sqrt.(LinearAlgebra.diag(StatsBase.vcov(m))) .* sqrt((n - k) / n)
    z  = β ./ se
    p  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    nm = replace.(string.(StatsBase.coefnames(m)), "(Intercept)" => "_cons")
    Printf.@printf("%-10s %13s %12s %8s %8s\n", dep, "Coef.", "Std. Err.", "z", "P>|z|")
    println(repeat("-", 55))
    for i in eachindex(β)
        Printf.@printf("%-10s %13.6g %12.6g %8.2f %8.3f\n", nm[i], β[i], se[i], z[i], p[i])
    end
    return (; β, se, z, p, names = nm)
end

# internal: integer label when a category value is a whole number, else its string
_ct_lab(c) = (c isa Integer || (c isa Real && isinteger(c))) ? string(Int(c)) : string(c)

"""
    ordered_classtable(β, τ, X, y_idx, cats;
                       link = z -> cdf(Normal(), z), rowname = "pclass") -> NamedTuple

Predicted-vs-actual classification table for an ordered-response model — Stata
`tab pclass y if e(sample)` after `oprobit` / `ologit`.

Forms the linear index `η = Xβ`, the predicted category probabilities

    P[i,j] = F(τⱼ − ηᵢ) − F(τⱼ₋₁ − ηᵢ)     (F(τ₀)=0, F(τ_J)=1)

for the `J = length(τ) + 1` ordered categories, assigns each row to its most
likely category `argmaxⱼ P[i,j]`, and cross-tabulates that predicted class
against the actual category index `y_idx` (values in `1:J`). `cats` are the `J`
outcome values, printed as the column headers. `link` is the standard CDF of the
latent error: the default standard normal gives ordered **probit**; pass a
logistic CDF `z -> 1/(1 + exp(-z))` for ordered **logit**.

Prints the `J×J` table with row/column `Total` margins, the overall
correctly-classified fraction `Σⱼ nⱼⱼ / N`, and each category's
diagonal/column-total ratio. Returns `(; pclass, ctab, P, correct)`.

`β`, `τ` and `X` are exactly the fields returned by [`stata_oprobit`](@ref) /
`stata_ologit` (`res.β`, `res.τ`, `res.X`); build `y_idx`/`cats` from the
estimation sample, e.g.
`cats = sort(unique(y)); y_idx = [findfirst(==(v), cats) for v in y]`.
"""
function ordered_classtable(β, τ, X, y_idx, cats;
                            link = z -> Distributions.cdf(Distributions.Normal(), z),
                            rowname::AbstractString = "pclass")
    η = X * β
    N = length(η); J = length(τ) + 1
    length(cats) == J ||
        error("length(cats)=$(length(cats)) must equal J = length(τ)+1 = $J")

    P = Matrix{Float64}(undef, N, J)
    for i in 1:N
        prev = 0.0
        for j in 1:(J - 1)
            c = link(τ[j] - η[i]); P[i, j] = c - prev; prev = c
        end
        P[i, J] = 1 - prev
    end
    pclass = [argmax(view(P, i, :)) for i in 1:N]
    ctab = [count(k -> pclass[k] == a && y_idx[k] == b, 1:N) for a in 1:J, b in 1:J]

    # ---- Stata `tab` layout (row-label width 10, category columns width 9) ----
    row(lbl, cells, tot) = lpad(lbl, 10) * " |" *
        join((lpad(string(c), 9) for c in cells), " ") * " |" * lpad(string(tot), 9)
    println(row(rowname, _ct_lab.(cats), "Total"))
    for a in 1:J
        println(row(string(a), ctab[a, :], sum(ctab[a, :])))
    end
    println(row("Total", [sum(ctab[:, b]) for b in 1:J], sum(ctab)))

    correct = sum(ctab[j, j] for j in 1:J) / N
    Printf.@printf("\ndi (%s)/%d = %.8f\n",
                   join((string(ctab[j, j]) for j in 1:J), "+"), N, correct)
    for j in 1:J
        cj = sum(ctab[:, j])
        Printf.@printf("di %d/%d = %.8f\n", ctab[j, j], cj, ctab[j, j] / cj)
    end
    return (; pclass, ctab, P, correct)
end

"""
    stata_fitstat(res, ll0, N; n_aux=1, error_var=res.sigma^2) -> NamedTuple

Post-estimation fit statistics for a latent-normal MLE model — Long & Freese's
`fitstat` after `tobit` / `intreg` and similar censored- or interval-normal
models. `res` is the fitted result (needs `res.β`, `res.ll`, and `res.X`), `ll0`
the intercept-only log-likelihood, and `N` the number of observations.

`n_aux` is the number of ancillary parameters counted alongside the regression
coefficients (default `1`, the error scale σ); `error_var` is the latent error
variance used by the McKelvey–Zavoina R² and the "variance of y*/error" rows
(default `res.sigma²`, i.e. tobit — pass `1.0` for probit, `π²/3` for logit).

The likelihood-based statistics (`D`, `LR`, McFadden, adjusted McFadden,
Cox–Snell `mlr2`, Cragg–Uhler `cu`, `aic`, `bic`, `bic'`) are general to any MLE;
only `mz`/`varys`/`vare` assume the latent-normal interpretation. Returns a
NamedTuple with fields consumed by [`fitstat_table`](@ref).
"""
function stata_fitstat(res, ll0, N; n_aux::Int = 1,
                       error_var::Real = res.sigma^2)
    kpar = length(res.β) + n_aux               # slopes + intercept + ancillary (σ)
    kx   = length(res.β) - 1                    # slopes only
    D    = -2 * res.ll
    LR   = 2 * (res.ll - ll0)
    xbv  = res.X * res.β
    varxb = Statistics.var(xbv)                 # Stata's fitstat uses the N-1 variance
    vare  = error_var
    (; ll0, ll = res.ll, D, dfD = N - kpar, LR, dfLR = kx,
       pLR   = 1 - Distributions.cdf(Distributions.Chisq(kx), LR),
       mcf   = 1 - res.ll / ll0,
       mcfa  = 1 - (res.ll - kpar) / ll0,
       mlr2  = 1 - exp(2 * (ll0 - res.ll) / N),
       cu    = (1 - exp(2 * (ll0 - res.ll) / N)) / (1 - exp(2 * ll0 / N)),
       mz    = varxb / (varxb + vare),
       varys = varxb + vare, vare,
       aic   = (D + 2 * kpar) / N, aicn = D + 2 * kpar,
       bic   = D - (N - kpar) * log(N), bicp = -LR + kx * log(N))
end

"""
    fitstat_table(current, saved; prob_lr_dif=nothing)

Print Long & Freese's `fitstat, dif()` Current / Saved / Difference comparison
of two [`stata_fitstat`](@ref) results (`current` is the model just fit, `saved`
the stored one). Every row's Difference is `current − saved`, except the deviance
`D` (Stata reports `saved − current`) and `Prob > LR`, whose Difference is the
p-value of the direct likelihood-ratio test between the two models — pass it as
`prob_lr_dif` (e.g. `stata_lrtest(...).p`); if omitted, `current.pLR − saved.pLR`
is shown.
"""
function fitstat_table(current, saved; prob_lr_dif = nothing)
    c = current; s = saved
    r3(lbl, cv, sv) = Printf.@printf("%-30s%13.3f%13.3f%13.3f\n", lbl, cv, sv, cv - sv)
    Printf.@printf("%-30s%13s%13s%13s\n", "", "Current", "Saved", "Difference")
    r3("Log-Lik Intercept Only:", c.ll0, s.ll0)
    r3("Log-Lik Full Model:",     c.ll,  s.ll)
    Printf.@printf("%-30s%9.3f(%d)%9.3f(%d)%9.3f(%d)\n", "D:",
                   c.D, c.dfD, s.D, s.dfD, s.D - c.D, s.dfD - c.dfD)
    Printf.@printf("%-30s%9.3f(%d)%11.3f(%d)%11.3f(%d)\n", "LR:",
                   c.LR, c.dfLR, s.LR, s.dfLR, c.LR - s.LR, c.dfLR - s.dfLR)
    Printf.@printf("%-30s%13.3f%13.3f%13.3f\n", "Prob > LR:", c.pLR, s.pLR,
                   prob_lr_dif === nothing ? c.pLR - s.pLR : prob_lr_dif)
    r3("McFadden's R2:",             c.mcf,   s.mcf)
    r3("McFadden's Adj R2:",         c.mcfa,  s.mcfa)
    r3("Maximum Likelihood R2:",     c.mlr2,  s.mlr2)
    r3("Cragg & Uhler's R2:",        c.cu,    s.cu)
    r3("McKelvey and Zavoina's R2:", c.mz,    s.mz)
    r3("Variance of y*:",            c.varys, s.varys)
    r3("Variance of error:",         c.vare,  s.vare)
    r3("AIC:",                       c.aic,   s.aic)
    r3("BIC:",                       c.bic,   s.bic)
    r3("BIC':",                      c.bicp,  s.bicp)
    return nothing
end
