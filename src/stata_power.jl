# ============================================================================
# stata_power.jl — Stata `power` Monte Carlo (Cameron & Trivedi ch12)
# ============================================================================

"""
    stata_power(numsims, numobs, b2_null, b2_alt, nominal; seed=10101) -> Float64

Stata-style `power` rclass program for OLS test-size and test-power Monte
Carlo. Each rep generates `numobs` observations from the DGP

    x ~ χ²(1),    y = 1 + b2_alt · x + (χ²(1) − 1)

then fits OLS, computes the two-sided t-test p-value for H0: β_x = b2_null,
and counts the share below `nominal`. When `b2_null == b2_alt` the result is
the empirical TEST SIZE; otherwise it is the empirical TEST POWER.

Uses `FixedEffectModels.reg` directly (no printed table per rep). Pass a
different `seed` for replication studies; default 10101 matches Stata's
`set seed 10101` convention used throughout the textbook.
"""
function stata_power(numsims::Int, numobs::Int, b2_null::Real, b2_alt::Real,
                     nominal::Float64; seed::Int = 10101)
    Random.seed!(seed)
    pvals = Vector{Float64}(undef, numsims)
    @inbounds for i in 1:numsims
        x  = rand(Distributions.Chisq(1), numobs)
        y  = 1 .+ b2_alt .* x .+ (rand(Distributions.Chisq(1), numobs) .- 1)
        df = DataFrames.DataFrame(x = x, y = y)
        m  = FixedEffectModels.reg(df, GLM.@formula(y ~ x))
        β  = StatsBase.coef(m)
        V  = StatsBase.vcov(m)
        ix = findfirst(==("x"), string.(StatsBase.coefnames(m)))
        t  = (β[ix] - b2_null) / sqrt(V[ix, ix])
        pvals[i] = 2 * (1 - Distributions.cdf(
            Distributions.TDist(StatsBase.dof_residual(m)), abs(t)))
    end
    return count(<(nominal), pvals) / numsims
end
