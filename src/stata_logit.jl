# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_logit — canonical `logit` estimator (Binomial GLM, LogitLink)
# This is the CANONICAL logit estimator in StatEcon; ch17/ch18 reuse it.
# --------------------------------------------------------------------------

"""
    stata_logit(df, formula; vce=:oim, level=0.95, quiet=false, cluster_var=nothing)
        -> NamedTuple

Stata-style `logit <depvar> <regs> [, vce(robust)]`. Fits a Binomial GLM
with `LogitLink` and prints Stata's exact `logit` output:

  - Header: "Logistic regression" + Number of obs / LR chi2(k) / Prob > chi2
    / Log likelihood / Pseudo R² (switches to `Wald chi2` / `Log
    pseudolikelihood` under `vce(robust)` / `vce(cluster)`).
  - Coefficient table in Stata-display order (slopes first, `_cons` last).

This is the canonical logit estimator in `StatEcon`. `StatsBase.coef(result.model)`
and `StatsBase.vcov(result.model)` work on the returned GLM object. Returns

    (; model, β, se, V, X, coefnames, ll, ll_null, pseudo_r2, LR, LR_p, n, k,
       β_glm, V_glm, X_glm, coefnames_glm)

where `β`, `se`, `V`, `X`, `coefnames` are in Stata-display order (slopes
first, `_cons` last, so `coefnames` ends `"_cons"`) and the `_glm`-suffixed
fields preserve GLM order. Mirrors `stata_probit` field-for-field so
downstream margins/dydx/lincom helpers can swap between the two.
"""
function stata_logit(df, formula; vce::Symbol=:oim, level::Float64=0.95,
                     quiet::Bool=false, cluster_var=nothing)
    needed = StatsModels.termvars(formula)
    cols   = cluster_var === nothing ? needed :
                                       vcat(needed, [Symbol(cluster_var)])
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    m  = GLM.glm(formula, dfc, Distributions.Binomial(), GLM.LogitLink())
    β  = GLM.coef(m); cn = GLM.coefnames(m)
    n  = Int(StatsBase.nobs(m)); k = length(β)
    yv = Float64.(GLM.response(m))
    Xm = GLM.modelmatrix(m)

    V = if vce == :robust
        # Robust sandwich: meat = Σ s_i s_iᵀ with score s_i = (y_i − μ_i)·x_i
        μ    = GLM.predict(m)
        s    = (yv .- μ) .* Xm
        meat = s' * s
        A    = Matrix(GLM.vcov(m))
        A * meat * A * (n / (n - 1))
    elseif vce == :cluster
        cluster_var === nothing &&
            error("vce = :cluster requires `cluster_var = :varname`")
        cl = dfc[!, Symbol(cluster_var)]
        μ  = GLM.predict(m)
        u  = (yv .- μ) .* Xm                # n×k per-obs scores
        meat = zeros(k, k)
        for g in unique(cl)
            sg = vec(sum(u[cl .== g, :], dims = 1))
            meat .+= sg * sg'
        end
        A = Matrix(GLM.vcov(m))
        G = length(unique(cl))
        # Stata's `logit, vce(cluster)` uses ONLY the G/(G−1) cluster
        # adjustment (no extra (n−1)/(n−k) factor).
        A * meat * A * (G / (G - 1))
    else
        Matrix(GLM.vcov(m))
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    p_z = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    ll = GLM.loglikelihood(m)
    m_null = GLM.glm(StatsModels.term(Symbol(formula.lhs)) ~ StatsModels.term(1),
                     dfc, Distributions.Binomial(), GLM.LogitLink())
    ll_null = GLM.loglikelihood(m_null)
    pseudo_r2 = 1 - ll / ll_null
    LR = 2 * (ll - ll_null)
    intercept_idx = findfirst(==("(Intercept)"), cn)
    slope_idx = setdiff(1:k, [intercept_idx])
    LR_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)), LR)

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w)
    end
    commafmt(num) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    # For robust / cluster vce, Stata switches to Wald chi² (computed on
    # the slopes using the robust V) and Log pseudolikelihood.
    is_rob = vce == :robust || vce == :cluster
    Wald, Wald_p = NaN, NaN
    if is_rob && !isempty(slope_idx)
        β_s = β[slope_idx]
        V_s = V[slope_idx, slope_idx]
        Wald   = β_s' * LinearAlgebra.inv(V_s) * β_s
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)), Wald)
    end

    if !quiet
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Logistic regression", "Number of obs", commafmt(n))
        if is_rob
            Printf.@printf("%56s%-13s = %6.2f\n", "",
                           "Wald chi2($(length(slope_idx)))", Wald)
            Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", Wald_p)
        else
            Printf.@printf("%56s%-13s = %6.2f\n", "",
                           "LR chi2($(length(slope_idx)))", LR)
            Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", LR_p)
        end
        ll_label = is_rob ? "Log pseudolikelihood" : "Log likelihood"
        ll_str = Printf.@sprintf("%s = %.4f", ll_label, ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad    = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad, r2_str)
        println()

        # Cluster sub-note (Stata: "(Std. err. adjusted for G clusters in <var>)").
        if vce == :cluster
            G = length(unique(dfc[!, Symbol(cluster_var)]))
            cl_str = Printf.@sprintf("(Std. err. adjusted for %d clusters in %s)",
                                     G, cluster_var)
            println(lpad(cl_str, 78))
        end

        println("-"^78)
        if is_rob
            println("             |               Robust")
        end
        se_label = is_rob ? "std. err." : "Std. err."
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       string(formula.lhs), se_label, 100*level)
        println("-"^13, "+", "-"^64)
        ord = intercept_idx === nothing ? collect(slope_idx) :
                                          vcat(slope_idx, [intercept_idx])
        for i in ord
            label = cn[i] == "(Intercept)" ? "_cons" : cn[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(β[i]; w=10), g9(se[i]; w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", p_z[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^78)
    end

    intercept_idx2 = findfirst(==("(Intercept)"), cn)
    slope_idx2     = setdiff(1:k, [intercept_idx2])
    ord2 = intercept_idx2 === nothing ? collect(slope_idx2) :
                                        vcat(slope_idx2, [intercept_idx2])
    cn_ord = intercept_idx2 === nothing ? cn[ord2] :
                                          vcat(cn[slope_idx2], "_cons")
    return (; model = m, β = β[ord2], se = se[ord2], V = V[ord2, ord2],
              X = Xm[:, ord2], coefnames = cn_ord,
              ll, ll_null, pseudo_r2, LR, LR_p, n, k,
              β_glm = β, V_glm = V, X_glm = Xm, coefnames_glm = cn)
end
