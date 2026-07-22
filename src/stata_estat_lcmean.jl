# ============================================================================
# stata_estat_lcmean.jl — Stata `estat lcmean` after `fmm` (C&T ch17)
# ============================================================================

"""
    stata_estat_lcmean(fmm_res; depvar="y", level=0.95) -> NamedTuple

Stata's `estat lcmean` after `fmm` — latent-class marginal means. For
each class c, reports the average predicted mean

    margin_c = (1/N) Σ_i μ_c(x_i),   μ_c = exp(x_i'β_c),

with a delta-method SE built from the fitted vcov `fmm_res.V` (only the
β_c block enters the gradient g_c = (1/N) Σ μ_c(x_i)·x_i; the class-logit
and other-component slots are zero). `fmm_res` is a `stata_fmm_poisson`
result (β in GLM column order, V over [β_1…β_C, γ_2…γ_C]). Prints the
Stata `Latent class marginal means` block. Returns `(; margins, ses)`.
"""
function stata_estat_lcmean(fmm_res; depvar::AbstractString = "y",
                           level::Float64 = 0.95)
    X = fmm_res.X; n, k = size(X)
    C = length(fmm_res.β_components)
    V = fmm_res.V
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    margins = Float64[]; ses = Float64[]
    for c in 1:C
        μc = exp.(X * fmm_res.β_components[c])
        m  = Statistics.mean(μc)
        gβ = vec(Statistics.mean(μc .* X, dims = 1))   # ∂m/∂β_c (length k)
        g  = zeros(size(V, 1))
        g[((c-1)*k + 1):(c*k)] .= gβ                   # place in β_c block
        se = sqrt(max(LinearAlgebra.dot(g, V * g), 0.0))
        push!(margins, m); push!(ses, se)
    end

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        su=sig; s=Printf.@sprintf("%.*g", su, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && su > 1; su-=1; s=Printf.@sprintf("%.*g", su, x); end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1.")); lpad(s, w)
    end
    println()
    println("Latent class marginal means")
    println("-"^78)
    Printf.@printf("%12s |     Margin   Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                   "", 100*level)
    println("-"^13, "+", "-"^64)
    for c in 1:C
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", "$c")
        m = margins[c]; s = ses[c]; z = m/s
        pp = 2*(1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       depvar, g9(m;w=10), g9(s;w=9),
                       Printf.@sprintf("%7.2f", z), Printf.@sprintf("%.3f", pp),
                       g9(m-crit*s;w=9), g9(m+crit*s;w=10))
    end
    println("-"^78)
    return (; margins, ses)
end
