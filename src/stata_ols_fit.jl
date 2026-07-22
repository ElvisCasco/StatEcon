# (deps provided by the StatEcon module)

"""
    stata_ols_fit(df, y, xs; level=0.95, vcov=nothing, weights=nothing) -> NamedTuple

Fit OLS via `FixedEffectModels.reg`, print the Stata regression table through
[`stata_regress`](@ref), and return a NamedTuple carrying the residuals and
the OLS log-likelihood — so the caller can reuse post-estimation quantities
(Stata's `e(ll)`, `predict, residuals`) without refitting.

Returns `(; model, β, residuals, ll, n, σ̂², coefnames)`. The log-likelihood
follows Stata's `e(ll)` convention

    ll = -n/2·log(2π σ̂²) - n/2,   σ̂² = RSS/n  (MLE).
"""
function stata_ols_fit(df::DataFrames.AbstractDataFrame, y::Symbol,
                       xs::AbstractVector{Symbol};
                       level::Float64 = 0.95,
                       vcov = nothing, weights = nothing)
    f = StatsModels.term(y) ~ sum(StatsModels.term.(xs))
    kw = weights === nothing ? (;) : (; weights = Symbol(weights))
    m = vcov === nothing ?
        FixedEffectModels.reg(df, f; kw...) :
        FixedEffectModels.reg(df, f, vcov; kw...)
    stata_regress(m)
    yvec = Float64.(_c16_rawval.(df[!, y]))
    rvec = yvec .- FixedEffectModels.predict(m, df)
    n    = Int(StatsBase.nobs(m))
    σ̂²   = sum(rvec.^2) / n
    ll   = -n/2 * log(2 * Base.pi * σ̂²) - n/2
    return (; model = m, β = StatsBase.coef(m), residuals = rvec,
              ll = ll, n = n, σ̂² = σ̂², coefnames = StatsBase.coefnames(m))
end
