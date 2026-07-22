"""
    stata_bsample(df; n=nrow(df), seed=nothing) -> DataFrame

Stata's `bsample` — observation-level resample with replacement. Returns
a new DataFrame with `n` rows drawn (with replacement) from `df`
(default `n = nrow(df)`). Pass `seed` for reproducibility; otherwise the
global RNG is used.

Note: Julia's `MersenneTwister` differs from Stata's MT64 RNG, so a fixed
seed will NOT reproduce Stata's exact bootstrap sample.
"""
function stata_bsample(df::DataFrames.AbstractDataFrame;
                       n::Int = DataFrames.nrow(df),
                       seed::Union{Int, Nothing} = nothing)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    idx = StatsBase.sample(rng, 1:DataFrames.nrow(df), n; replace = true)
    return df[idx, :]
end
