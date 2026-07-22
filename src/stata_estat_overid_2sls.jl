# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_estat_overid_2sls — `estat overid` (Sargan) after `ivregress 2sls`
# --------------------------------------------------------------------------

"""
    stata_estat_overid_2sls(df; depvar, exog_vars, endog, instruments,
                            quiet=false) -> NamedTuple

Stata's `estat overid` after `ivregress 2sls`. Computes the Sargan / Hansen
score test of overidentifying restrictions:

  - Fit 2SLS of `depvar` on `exog_vars` with `(endog = instruments)`, get
    residuals `û`.
  - Auxiliary OLS  `û ~ exog + instruments`  (full instrument set).
  - Statistic = n · R²  ~  χ²(L − K),  with L = # excluded instruments and
    K = # endogenous regressors (a single endogenous regressor here).

Prints Stata's standard block:

      Test of overidentifying restrictions:

      Score chi2(L−K)          =  <stat>  (p = <p>)

Returns `(; chi2, df, pvalue)`.
"""
function stata_estat_overid_2sls(df; depvar::Symbol,
                                 exog_vars::AbstractVector{Symbol},
                                 endog::Symbol,
                                 instruments::AbstractVector{Symbol},
                                 quiet::Bool = false)
    needed = unique(vcat(depvar, endog, exog_vars, instruments))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    keep = trues(DataFrames.nrow(dfc))
    for c in needed
        col = dfc[!, c]
        eltype(col) <: Real || continue
        for i in eachindex(col)
            keep[i] &= isfinite(col[i])
        end
    end
    dfc = dfc[keep, :]

    # 2SLS: depvar ~ exog + (endog ~ instruments) via FixedEffectModels.
    fml_iv = StatsModels.term(depvar) ~
             sum(StatsModels.term.(exog_vars)) +
             (StatsModels.term(endog) ~ sum(StatsModels.term.(instruments)))
    m = FixedEffectModels.reg(dfc, fml_iv)

    yhat = FixedEffectModels.predict(m, dfc)
    yvec = Float64.(dfc[!, depvar])
    kk   = .!ismissing.(yhat) .& isfinite.(yvec) .& isfinite.(yhat)
    uhat = yvec[kk] .- yhat[kk]

    # Auxiliary regression of û on (exog + instruments + intercept).
    n_aux = length(uhat)
    Z = hcat(ones(n_aux),
             [Float64.(dfc[kk, v]) for v in exog_vars]...,
             [Float64.(dfc[kk, v]) for v in instruments]...)
    γ̂  = Z \ uhat
    res_aux = uhat .- Z * γ̂
    ss_tot  = sum(uhat .^ 2)                 # uncentered (no demean — matches Sargan)
    R2  = 1 - sum(res_aux .^ 2) / ss_tot
    chi2  = n_aux * R2
    df_st = length(instruments) - 1   # L_excl − K_endog (single endogenous regressor)
    pv    = 1 - Distributions.cdf(Distributions.Chisq(df_st), chi2)

    if !quiet
        # Strip leading 0 for |x|<1 in Stata's display style
        g(x) = (s = Printf.@sprintf("%.6g", x);
                0 < abs(x) < 1 ? replace(s, r"^(-?)0\." => s"\1.") : s)
        println()
        println("  Test of overidentifying restrictions:")
        println()
        Printf.@printf("  Score chi2(%d)          =  %s  (p = %.4f)\n",
                       df_st, g(chi2), pv)
    end
    return (; chi2, df = df_st, pvalue = pv)
end
