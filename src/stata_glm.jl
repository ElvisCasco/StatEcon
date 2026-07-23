# ============================================================================
# stata_glm.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_glm(df, formula; family=:poisson, link=:log, vce=:robust, level=0.95)

Stata's `glm y x…, family(...) link(...) vce(robust)`. Prints the GLM-format
header (deviance, Pearson χ², AIC, BIC, variance/link function) followed by a
coefficient table.

  - `family` : `:poisson`, `:binomial`, or `:gaussian`.
  - `link`   : `:log`, `:identity`, `:logit`, or `:probit`.
  - `vce`    : `:robust` (HC1, poisson only) or otherwise the GLM vcov.

Returns a NamedTuple: `model, β, se, V, ll, deviance, pearson, aic, bic, n`
(GLM coefficient order).
"""
function stata_glm(df, formula; family::Symbol=:poisson, link::Symbol=:log,
                    vce::Symbol=:robust, level::Float64=0.95)
    needed = StatsModels.termvars(formula)
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    fam = family == :poisson  ? Distributions.Poisson() :
          family == :binomial ? Distributions.Binomial() :
          family == :gaussian ? Distributions.Normal()   :
          error("family=$family not supported")
    lnk = link == :log     ? GLM.LogLink()    :
          link == :identity ? GLM.IdentityLink() :
          link == :logit   ? GLM.LogitLink()  :
          link == :probit  ? GLM.ProbitLink() :
          error("link=$link not supported")

    m = GLM.glm(formula, dfc, fam, lnk)
    β  = GLM.coef(m)
    cn = GLM.coefnames(m)
    n  = Int(StatsBase.nobs(m))
    yv = Float64.(GLM.response(m))
    μ  = GLM.predict(m)
    Xm = GLM.modelmatrix(m)
    k  = size(Xm, 2)
    df_resid = n - k

    if vce == :robust && family == :poisson
        H = Xm' * LinearAlgebra.Diagonal(μ) * Xm
        meat = Xm' * LinearAlgebra.Diagonal((yv .- μ).^2) * Xm
        Hinv = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H)))
        # Stata's `glm, vce(robust)` uses the n/(n-1) finite-sample correction
        # (same as `poisson, vce(robust)`), NOT n/(n-k). Verified on Wooldridge
        # Ex. 19.1: n/(n-1) gives educ SE .0025918, matching Stata exactly, where
        # n/(n-k) gave .0025938 (+0.08%).
        V = Hinv * meat * Hinv * (n / (n - 1))
    else
        V = Matrix(GLM.vcov(m))
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    p  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    deviance_val = GLM.deviance(m)
    pearson = family == :poisson ? sum((yv .- μ).^2 ./ μ) : NaN
    dev_per_df  = deviance_val / df_resid
    pear_per_df = pearson / df_resid

    ll  = GLM.loglikelihood(m)
    aic = (-2 * ll + 2 * k) / n
    bic = deviance_val - df_resid * log(n)

    var_desc  = family == :poisson ? "V(u) = u" :
                family == :binomial ? "V(u) = u·(1-u/n)" : "V(u) = 1"
    link_desc = link == :log ? "g(u) = ln(u)" :
                link == :identity ? "g(u) = u" :
                link == :logit ? "g(u) = ln(u/(1-u))" :
                link == :probit ? "g(u) = invΦ(u)" : "g(u) = ?"
    fam_label = family == :poisson ? "[Poisson]" :
                family == :binomial ? "[Binomial]" : "[$family]"
    lnk_label = link == :log ? "[Log]" :
                link == :identity ? "[Identity]" :
                link == :logit ? "[Logit]" :
                link == :probit ? "[Probit]" : "[$link]"

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return lpad(s, w)
    end
    function commafmt(num)
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        return (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    println()
    Printf.@printf("%-50s%-15s = %10s\n", "Generalized linear models",
                   "Number of obs", commafmt(n))
    Printf.@printf("%-50s%-15s = %10d\n", "Optimization     : ML",
                   "Residual df", df_resid)
    Printf.@printf("%-50s%-15s = %10d\n", "", "Scale parameter", 1)
    Printf.@printf("%-15s = %12.5f                   %-15s = %10.5f\n",
                   "Deviance", deviance_val, "(1/df) Deviance", dev_per_df)
    Printf.@printf("%-15s = %12.5f                   %-15s = %10.5f\n",
                   "Pearson", pearson, "(1/df) Pearson", pear_per_df)
    println()
    Printf.@printf("%-50s%s\n", "Variance function: $(var_desc)", fam_label)
    Printf.@printf("%-50s%s\n", "Link function    : $(link_desc)", lnk_label)
    println()
    Printf.@printf("%50s%-15s = %10.6f\n", "", "AIC", aic)
    Printf.@printf("Log pseudolikelihood = %.5f%18s%-15s = %10.3f\n",
                   ll, "", "BIC", bic)
    println()

    println("-"^78)
    if vce == :robust
        println("             |               Robust")
    end
    Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [95%% conf. interval]\n",
                   string(formula.lhs))
    println("-"^13, "+", "-"^64)
    intercept_idx = findfirst(==("(Intercept)"), cn)
    slope_idx = setdiff(1:length(β), [intercept_idx])
    for i in slope_idx
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       cn[i], g9(β[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       g9(ci_lo[i]; w=10), g9(ci_hi[i]; w=10))
    end
    if intercept_idx !== nothing
        i = intercept_idx
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "_cons", g9(β[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       g9(ci_lo[i]; w=10), g9(ci_hi[i]; w=10))
    end
    println("-"^78)

    return (; model=m, β, se, V, ll, deviance=deviance_val, pearson, aic, bic, n)
end
