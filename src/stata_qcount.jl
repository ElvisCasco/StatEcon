## Machado–Santos Silva (2005) jittered quantile count regression
## and marginal-effects display for exp-link count models.

# ────────────────────────────────────────────────────────────────────────
# `stata_qcount` — SSC `qcount y xs…, q(τ) rep(B)` (Miranda 2007)
#
#   for r = 1, …, reps:
#     1. paired-bootstrap resample rows
#     2. jitter:  y*_i = y_i + U(0,1)
#     3. IRLS quantile regression at τ on  log(y*)  ⇒  β̂_r
#   β̂ = mean(β̂_r);  V̂ = cov(β̂_r)
#
# Both the sampling (paired bootstrap) and jittering uncertainty are
# captured in V̂, matching Stata's `qcount`.
# ────────────────────────────────────────────────────────────────────────

# Local IRLS: identical to `_qr_irls` in stata_qreg.jl but with tighter defaults
# suitable for the inner loop of `stata_qcount`.
function _qc_irls(X::AbstractMatrix, Y::AbstractVector, τ::Real;
                  max_iter::Int = 2000, tol::Float64 = 1e-10)
    n     = length(Y)
    β     = X \ Y
    one_v = ones(n)
    δ     = 1e-2
    for _ in 1:max_iter
        e    = Y .- X * β
        δ    = max(min(δ, 0.5 * Statistics.median(abs.(e)) + 1e-12), 1e-10)
        w    = 1.0 ./ max.(abs.(e), δ)
        rhs  = X' * (w .* Y) .+ (2τ - 1) .* (X' * one_v)
        XtWX = X' * (w .* X)
        β_nw = XtWX \ rhs
        dif  = sqrt(sum((β_nw .- β).^2)) / max(1.0, sqrt(sum(β.^2)))
        β    = β_nw
        dif < tol && break
    end
    return β
end

"""
    stata_qcount(df, y, xs; τ=0.5, reps=500, seed=nothing, level=0.95)

Stata-style `qcount y xs…, q(τ) rep(B)` — Machado–Santos Silva (2005)
quantile count regression. Each replicate combines a paired bootstrap of
rows with a fresh U(0,1) jitter of the count response, then fits IRLS
quantile regression on `log(y + U)`. β̂ = mean over reps, V̂ = cov over
reps. Prints Stata's Bootstrap coefficient block; returns
`(; β, V, se, t, p, ci_lo, ci_hi, coefnames, n, reps, τ, df_resid)`.
"""
function stata_qcount(df, y, xs::AbstractVector;
                      τ::Real = 0.5, reps::Int = 500,
                      seed::Union{Int,Nothing} = nothing,
                      level::Float64 = 0.95)
    ys  = Symbol(y)
    xsv = [Symbol(v) for v in xs]
    d   = DataFrames.dropmissing(df, unique(vcat(ys, xsv)))
    n   = DataFrames.nrow(d)
    Y   = Float64.(_sm_rawval.(d[!, ys]))
    X   = hcat([Float64.(_sm_rawval.(d[!, v])) for v in xsv]..., ones(n))
    k   = size(X, 2)

    seed !== nothing && Random.seed!(seed)
    β_draws = Matrix{Float64}(undef, reps, k)
    for r in 1:reps
        idx = StatsBase.sample(1:n, n; replace = true)
        Xb  = X[idx, :]
        Yb  = Y[idx]
        β_draws[r, :] = _qc_irls(Xb, log.(Yb .+ rand(n)), τ)
    end

    β  = vec(Statistics.mean(β_draws; dims = 1))
    V  = Statistics.cov(β_draws)
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    α     = 1 - level
    dofv  = n - k
    t_st  = β ./ se
    tcrit = Distributions.quantile(Distributions.TDist(dofv), 1 - α/2)
    pvals = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(dofv), abs.(t_st)))
    ci_lo = β .- tcrit .* se
    ci_hi = β .+ tcrit .* se

    println()
    title = Printf.@sprintf("Jittered count-quantile regression (τ = %.2f)", τ)
    Printf.@printf("%s%*sNumber of obs = %10s\n",
                   title, max(0, 52 - length(title)), "", _qr_fmtn(n))
    Printf.@printf("  No. jittered samples = %d\n\n", reps)

    nms = vcat(string.(xsv), "_cons")
    println("-"^78)
    Printf.@printf("%12s | %22s\n", "", "Bootstrap")
    Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                   string(ys), "Coefficient", "std. err.", "t", "P>|t|",
                   round(Int, 100*level))
    println("-"^13, "+", "-"^64)
    for i in 1:k
        Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                       nms[i], _qr_g(β[i], 10), _qr_g(se[i], 9),
                       t_st[i], pvals[i],
                       _qr_g(ci_lo[i], 11), _qr_g(ci_hi[i], 10))
    end
    println("-"^78)

    return (; β, V, se, t = t_st, p = pvals, ci_lo, ci_hi,
              coefnames = nms, n, reps, τ, df_resid = dofv)
end

# ────────────────────────────────────────────────────────────────────────
# `stata_margins_count` — marginal effects for exp-link count models.
# Accepts either a GLM-style fit or a NamedTuple with (β, V, coefnames)
# (e.g. output of `stata_qcount` / `stata_qreg`). Prints MEM (`atmean=true`)
# or AME (`atmean=false`) with delta-method SEs and (optionally) 1.var
# binary-treatment discrete differences.
# ────────────────────────────────────────────────────────────────────────

_mc_is_cons(n::AbstractString) = n == "(Intercept)" || n == "_cons"

"""
    stata_margins_count(model, df; binary=Symbol[], atmean=true, level=0.95)

Marginal effects for exp-link (log-link) count models — Poisson, negative
binomial, or a fitted `stata_qcount` NamedTuple. When `atmean=true` (default),
evaluates the effect at the mean of the covariates (MEM); when `atmean=false`,
returns average marginal effects (AME) over the sample. Any variable listed in
`binary` uses the discrete difference μ(x=1) − μ(x=0) with the corresponding
delta-method gradient. Row labels for such variables are prefixed `1.var`,
mirroring Stata's `margins, dydx(*)` output.
"""
function stata_margins_count(model, df;
                             binary::AbstractVector = Symbol[],
                             atmean::Bool = true,
                             level::Float64 = 0.95)
    β  = hasproperty(model, :β) ? Float64.(model.β)         : StatsBase.coef(model)
    V  = hasproperty(model, :V) ? Matrix{Float64}(model.V)   : StatsBase.vcov(model)
    cn = hasproperty(model, :coefnames) ? String.(model.coefnames) :
                                          string.(StatsBase.coefnames(model))

    binary_str = Set(String.(string.(binary)))
    nms        = [n for n in cn if !_mc_is_cons(n)]

    rows = NamedTuple[]
    if atmean
        means_d = Dict{String,Float64}()
        for n in nms
            hasproperty(df, Symbol(n)) ||
                error("Variable $n not in dataframe")
            means_d[n] = Statistics.mean(_sm_rawval.(skipmissing(df[!, Symbol(n)])))
        end
        x̄  = [_mc_is_cons(n) ? 1.0 : means_d[n] for n in cn]
        μ̄  = exp(LinearAlgebra.dot(x̄, β))

        for (j, n) in enumerate(cn)
            _mc_is_cons(n) && continue
            if n in binary_str
                x1 = copy(x̄);  x1[j] = 1.0
                x0 = copy(x̄);  x0[j] = 0.0
                μ1 = exp(LinearAlgebra.dot(x1, β))
                μ0 = exp(LinearAlgebra.dot(x0, β))
                dydx = μ1 - μ0
                g    = x1 .* μ1 .- x0 .* μ0
                label = "1." * n
            else
                dydx = β[j] * μ̄
                g    = μ̄ .* (x̄ .* β[j])
                g[j] += μ̄
                label = n
            end
            se = sqrt(max(g' * V * g, 0.0))
            zv = se > 0 ? dydx / se : NaN
            pv = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(zv)))
            zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
            lo = dydx - zcrit*se;  hi = dydx + zcrit*se
            push!(rows, (; Variable = label, dydx, StdErr = se, z = zv,
                          p = pv, CI_low = lo, CI_high = hi))
        end
    else
        cleaned = DataFrames.dropmissing(df,
            [Symbol(n) for n in nms if hasproperty(df, Symbol(n))])
        X = zeros(DataFrames.nrow(cleaned), length(cn))
        for (j, n) in enumerate(cn)
            X[:, j] = _mc_is_cons(n) ? ones(DataFrames.nrow(cleaned)) :
                      Float64.(_sm_rawval.(cleaned[!, Symbol(n)]))
        end
        μ_vec = hasproperty(model, :β) ? exp.(X * β) :
                Float64.(GLM.predict(model, cleaned))
        μ̄_ = Statistics.mean(μ_vec)
        for (j, n) in enumerate(cn)
            _mc_is_cons(n) && continue
            dydx = β[j] * μ̄_
            g = (β[j] .* (X' * μ_vec) ./ DataFrames.nrow(cleaned))
            g[j] += μ̄_
            se = sqrt(max(g' * V * g, 0.0))
            zv = se > 0 ? dydx / se : NaN
            pv = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(zv)))
            zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
            lo = dydx - zcrit*se;  hi = dydx + zcrit*se
            push!(rows, (; Variable = n, dydx, StdErr = se, z = zv,
                          p = pv, CI_low = lo, CI_high = hi))
        end
    end

    mfx_df = DataFrames.DataFrame(rows)
    n_obs  = hasproperty(model, :n) ? Int(model.n) :
             (try Int(StatsBase.nobs(model)) catch; DataFrames.nrow(df) end)
    mode_l = atmean ? "MEM (atmean)" : "AME"
    header = "Conditional marginal effects — $mode_l"
    println(header, " "^max(0, 50 - length(header)),
            "Number of obs = $n_obs")
    println()

    lvl = round(Int, 100 * level)
    println("-"^80)
    Printf.@printf("%12s | %11s  %10s  %6s  %5s   [%d%% conf. interval]\n",
                   "Variable", "dy/dx", "std. err.", "z", "P>|z|", lvl)
    println("-"^13, "+", "-"^66)
    for r in rows
        Printf.@printf("%12s | %11.7f  %10.7f  %6.2f  %6.3f  %10.7f  %10.7f\n",
                       r.Variable, r.dydx, r.StdErr, r.z, r.p,
                       r.CI_low, r.CI_high)
    end
    println("-"^80)
    return (; mfx = rows, mfx_df)
end
