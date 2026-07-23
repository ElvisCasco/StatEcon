"""
    _probit_oim_vcov(X, y, beta) -> Matrix

Inverse of the OBSERVED information for a probit fit — the variance Stata's
`vce(oim)` default reports. `GLM.vcov` gives the EXPECTED (IRLS/Fisher)
information instead; for the non-canonical probit link the two differ, which
left probit standard errors about 1-2% above Stata's.

With `q = 2y-1`, `z = q*(X*beta)` and `lambda = q*phi(z)/Phi(z)`, the negative
Hessian of the log-likelihood is `sum_i lambda_i (lambda_i + eta_i) x_i x_i'`.

Shared by `stata_probit` and the stage-1 probit of `stata_heckman_twostep` so
the two cannot drift apart.
"""
function _probit_oim_vcov(X::AbstractMatrix, y::AbstractVector, β::AbstractVector)
    η  = X * β
    q  = 2 .* y .- 1
    z  = q .* η
    Φz = Distributions.cdf.(Distributions.Normal(), z)
    φz = Distributions.pdf.(Distributions.Normal(), z)
    λ  = q .* (φz ./ max.(Φz, 1e-12))
    w  = λ .* (λ .+ η)
    return LinearAlgebra.inv(LinearAlgebra.Symmetric(X' * (w .* X)))
end

"""
    stata_probit(df, formula; vce=:oim, level=0.95, quiet=false) -> NamedTuple

Stata-style `probit <depvar> <regs> [, vce(robust)]`. Fits a Binomial GLM
with `ProbitLink` and prints Stata's exact `probit` output:

  - Header: Number of obs / LR chi2(k) / Prob > chi2 / Log likelihood / Pseudo R²
    (NB: `LR chi2`, NOT `Wald chi2` — Stata uses 2·(ll − ll_null) for the
    overall fit statistic in `probit` even with `vce(robust)`.)
  - Coefficient table in Stata-display order (slopes first, `_cons` last).

This is the canonical probit estimator in `StatEcon`; other chapters
(e.g. Heckman two-step in ch16, ordered/multinomial helpers in ch14) reuse
the fitted probit linear index via the returned `model` / `β_glm`.

`StatsBase.coef(result.model)` and `StatsBase.vcov(result.model)` work on the
returned GLM object. Returns

    (; model, β, se, V, X, coefnames, ll, ll_null, pseudo_r2, LR, LR_p, n, k,
       β_glm, V_glm, X_glm, coefnames_glm)

where `β`, `se`, `V`, `X`, `coefnames` are in Stata-display order (slopes
first, `_cons` last) and the `_glm`-suffixed fields preserve GLM order.
"""
function stata_probit(df, formula; vce::Symbol=:oim, level::Float64=0.95,
                      quiet::Bool=false)
    needed = StatsModels.termvars(formula)
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    # Drop any rows with non-finite values (NaN/Inf in computed columns
    # like `linc = log(hhincome)` would crash GLM.jl's IRLS at the
    # `isfinite(dev)` assertion). `dropmissing` only handles `Missing`.
    keep = trues(DataFrames.nrow(dfc))
    for c in needed
        col = dfc[!, c]
        eltype(col) <: Real || continue
        for i in eachindex(col)
            keep[i] &= isfinite(col[i])
        end
    end
    dfc = dfc[keep, :]
    # Fit probit. Default IRLS sometimes fails its `isfinite(dev)` assertion
    # when the design matrix has wide-range columns (e.g. `age2`); in that
    # case we warm-start from a logit fit's β rescaled by ≈1/1.81 (the
    # asymptotic logit→probit coefficient ratio sqrt(π/3)).
    m = try
        GLM.glm(formula, dfc, Distributions.Binomial(), GLM.ProbitLink())
    catch err
        if err isa AssertionError && occursin("isfinite", string(err))
            m_logit = GLM.glm(formula, dfc, Distributions.Binomial(), GLM.LogitLink())
            β_init  = GLM.coef(m_logit) ./ 1.81
            GLM.glm(formula, dfc, Distributions.Binomial(), GLM.ProbitLink();
                    start = β_init)
        else
            rethrow(err)
        end
    end
    β  = GLM.coef(m); cn = GLM.coefnames(m)
    n  = Int(StatsBase.nobs(m)); k = length(β)
    yv = Float64.(GLM.response(m))
    Xm = GLM.modelmatrix(m)

    A_oim = _probit_oim_vcov(Xm, yv, β)

    V = if vce == :robust
        # Robust sandwich: V = A · meat · A · (n/(n-1)) with A the observed-
        # information inverse and meat = Σ s_i s_iᵀ, the per-obs score being
        # s_i = ((y_i − μ_i) / [μ_i(1−μ_i)]) · φ(η_i) · x_i.
        η  = Xm * β                                 # linear index
        μ  = GLM.predict(m)
        ϕ  = Distributions.pdf.(Distributions.Normal(), η)
        u  = (yv .- μ) ./ (μ .* (1 .- μ)) .* ϕ
        s  = u .* Xm
        meat = s' * s
        A_oim * meat * A_oim * (n / (n - 1))
    else
        Matrix(A_oim)
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    p_z = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    ll = GLM.loglikelihood(m)
    m_null = GLM.glm(StatsModels.term(Symbol(formula.lhs)) ~ StatsModels.term(1),
                     dfc, Distributions.Binomial(), GLM.ProbitLink())
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

    if !quiet
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Probit regression", "Number of obs", commafmt(n))
        Printf.@printf("%56s%-13s = %6.2f\n", "",
                       "LR chi2($(length(slope_idx)))", LR)
        Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", LR_p)
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad    = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad, r2_str)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(formula.lhs), 100*level)
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
