# (deps provided by the StatEcon module)

"""
    stata_sktest(x; quiet=false, name="x") -> NamedTuple
    stata_sktest(df, var::Symbol; kwargs...)

Stata `sktest <var>` — the D'Agostino / Royston joint skewness-kurtosis test
for normality on a numeric vector (or a DataFrame column). Uses the asymptotic
standard errors

    z_skew = skewness · √(N/6)
    z_kurt = (kurtosis − 3) · √(N/24)
    χ²(2)  = z_skew² + z_kurt²

(the Royston small-sample correction is omitted; this agrees with Stata to
≤ 3 decimals for N ≳ 1000). Returns
`(; skewness, kurtosis, z_sk, z_kt, p_sk, p_kt, χ², p)`.
"""
function stata_sktest(x::AbstractVector{<:Real}; quiet::Bool = false,
                      name::AbstractString = "x")
    n  = length(x); μ = Statistics.mean(x)
    m2 = Statistics.mean((x .- μ).^2)
    m3 = Statistics.mean((x .- μ).^3)
    m4 = Statistics.mean((x .- μ).^4)
    g1 = m3 / m2^(3/2)
    g2 = m4 / m2^2 - 3.0
    z_sk = g1 / sqrt(6 / n)
    z_kt = g2 / sqrt(24 / n)
    p_sk = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_sk)))
    p_kt = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_kt)))
    χ²   = z_sk^2 + z_kt^2
    p    = 1 - Distributions.cdf(Distributions.Chisq(2), χ²)
    if !quiet
        println()
        println("Skewness and kurtosis tests for normality ($name)")
        println("------------------------------------------------------------")
        Printf.@printf("Skewness = %.6f   z = %6.2f   Pr(Skewness) = %.4f\n",
                       g1, z_sk, p_sk)
        Printf.@printf("Kurtosis = %.6f   z = %6.2f   Pr(Kurtosis) = %.4f\n",
                       g2 + 3, z_kt, p_kt)
        Printf.@printf("Joint chi2(2) = %.2f   Prob > chi2 = %.4f\n", χ², p)
    end
    return (; skewness = g1, kurtosis = g2 + 3, z_sk, z_kt, p_sk, p_kt, χ², p)
end

function stata_sktest(df::DataFrames.AbstractDataFrame, var::Symbol; kwargs...)
    x = Float64.(_c16_rawval.(collect(skipmissing(df[!, var]))))
    return stata_sktest(x; name = string(var), kwargs...)
end
