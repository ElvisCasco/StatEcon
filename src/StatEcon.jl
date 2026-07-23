"""
    StatEcon

Julia equivalents of common Stata commands, with Stata-style printed output:

  - `stata_regress`   -> `regress y x, vce(robust)` / `xtreg ..., fe`
  - `stata_regress_cluster` -> `regress y x [pw=w], vce(cluster id)`
  - `stata_tabulate`  -> `tabulate var`
  - `stata_summarize` -> `summarize var [if] [, detail]` (varlist globs supported)
  - `stata_ttest`     -> `ttest var, by(group)`
  - `stata_test`      -> `test <varlist>` after a regression (joint Wald F)
  - `stata_margins`   -> `margins, dydx(var)` / `margins, eyex(var) atmean`
  - `stata_mean`      -> `mean var`
  - `stata_svy_mean`  -> `svy: mean var`
  - `stata_svy_regress` -> `svy: regress y x1 x2`
  - `stata_sureg`     -> `sureg (y1: x1s) (y2: x2s) ...` (SUR by FGLS)
  - `stata_test_sureg` -> `test <varlist>` after `sureg`
  - `stata_reg3`      -> `reg3 (eq1) (eq2) ...` (3SLS systems)
  - `stata_ivregress_2sls` -> `ivregress 2sls y (endog = iv) exog`
  - `stata_ivregress_gmm`  -> `ivregress gmm  y (endog = iv) exog, wmatrix(...)`
  - `stata_liml`      -> `ivregress liml y (endog = iv) exog`
  - `stata_jive`      -> Jackknife IV (JIVE1)
  - `stata_estimates_table` -> `estimates table est1 est2 ..., b se`
  - `estat_endogenous` -> `estat endogenous` (Durbin-Wu-Hausman)
  - `estat_overid`     -> `estat overid` (Hansen J)
  - `estat_firststage` -> `estat firststage[, forcenonrobust all]`
  - `stata_correlate`  -> `correlate varlist`
  - `stata_treatreg`   -> `treatreg y indepvars, treat(D = varlist)`
  - `stata_qreg`       -> `qreg y x1 x2, quantile(τ)`
  - `stata_sqreg`      -> `sqreg y x1 x2, q(τ1 τ2 …) reps(B)`
  - `stata_qcount`     -> `qcount y x1 x2, q(τ) rep(B)` (Machado–Santos Silva)
  - `stata_margins_count` -> `margins, dydx(*)` for exp-link count models
  - `estat_hettest`   -> `estat hettest` (Breusch-Pagan / Cook-Weisberg)
  - `estat_imtest`    -> `estat imtest` (IM test, C&T decomposition)
  - `stata_boxcox`    -> `boxcox y x1 x2 ...` (LHS-only MLE)
  - `stata_xtdescribe` -> `xtdescribe` (unbalanced-panel pattern table)
  - `stata_xtsum`     -> `xtsum var` (overall / between / within summary)
  - `stata_xttab`     -> `xttab var` (panel one-way tabulation)
  - `stata_xttrans`   -> `xttrans var, freq` (state-transition frequencies)
  - `stata_matlist`   -> `matlist M` (pretty-print a small matrix)
  - `stata_xtreg_pa`  -> `xtreg y x, pa corr(exchangeable|ar1|...) vce(robust)`
  - `stata_xtreg_re`  -> `xtreg y x, re` (Swamy–Arora random-effects GLS)
  - `stata_xtreg_re_print` -> Stata-format printer for `stata_xtreg_re` results
  - `stata_hausman`   -> `hausman fe re` (Hausman specification test)
  - `stata_areg`      -> `areg y x, absorb(id) [vce(robust|cluster id)]`
  - `stata_xtregar`   -> `xtregar y x, re|fe` (Baltagi–Wu AR(1) panel)
  - `stata_xtscc`     -> Driscoll–Kraay HAC panel SE
  - `stata_xtgls`     -> `xtgls y x, panels(...) corr(...) igls`
  - `ereturn_list`    -> `ereturn list` after an e-class command
  - `psmatch2`        -> `psmatch2 treat covars, outcome(y) logit ate`
  - `teffects_ipw`    -> `teffects ipw (y) (treat covars)`
  - `kdensity`        -> `kdensity var` (Gaussian KDE, Silverman bandwidth)

Nonlinear & limited-dependent-variable models (Cameron & Trivedi ch10–ch18):

  - `stata_poisson` / `stata_glm` / `stata_nbreg` / `stata_gnbreg`   (count / GLM)
  - `stata_probit` / `stata_logit` / `stata_hetprob`                 (binary outcome)
  - `stata_ivprobit` / `stata_ivprobit_twostep`                      (endogenous probit)
  - `stata_tobit` / `stata_heckman` / `stata_heckman_twostep`        (censoring / selection)
  - `stata_mlogit` / `stata_mprobit` / `stata_ologit` / `stata_biprobit`
  - `stata_nlogit` / `stata_asclogit` / `stata_clogit` / `stata_asmprobit` / `stata_mixlogit`
  - `stata_ztnb` / `stata_zinb` / `stata_fmm_poisson` / `stata_fmm_nbreg`  (truncated / zero-inflated / finite-mixture)
  - `stata_poisson_gmm` / `stata_gmm_poisson` / `stata_nl_exp`       (GMM / NLS exponential)
  - `stata_nlcom` / `stata_lincom` / `stata_testnl` / `stata_lrtest` / `stata_power`  (post-estimation / testing)
  - `stata_margins_at` / `stata_margins_dydx` / `stata_mfx` / `stata_margins_tobit` / `stata_margins_mlogit`
  - `stata_estat_ic` / `stata_estat_classification` / `stata_estat_gof` / `stata_estat_lcmean` / `stata_estat_overid_gmm`

Panel models (ch08–ch09, ch18):

  - `stata_xtivreg` / `stata_xtivreg_fe` / `stata_hausman_taylor`    (panel IV)
  - `stata_arellano_bond` / `stata_xtdpdsys` / `stata_estat_abond`   (dynamic panel GMM)
  - `stata_xtmixed` / `stata_gllamm` / `stata_recovariance`          (mixed / multilevel)
  - `stata_xtlogit_pa|re|fe` / `stata_xtpoisson_pa|re` / `stata_xtnbreg_pa|re|fe`
  - `stata_xttobit_re` / `stata_xtmelogit`                           (nonlinear panel)

Every chapter of Cameron & Trivedi, *Microeconometrics Using Stata*, is reproduced
using these commands under `examples/Cameron_Trivedi/ch01…ch18`.

Calls are normally written module-qualified, e.g. `StatEcon.stata_regress(...)`.
"""
module StatEcon

using DataFrames, ReadStatTables, GLM, FixedEffectModels, StatsModels, Statistics, Printf
using StatsBase: nobs, dof, dof_residual, r2, adjr2, deviance, coef, vcov, coefnames
import StatsBase
using LinearAlgebra
import Distributions
using Random
using PrecompileTools

include("stata_tabulate.jl")
include("stata_summarize.jl")
include("kdensity.jl")
include("psmatch2.jl")
include("teffects_ipw.jl")
include("stata_regress.jl")
include("ereturn_list.jl")
include("stata_ttest.jl")
include("stata_test.jl")
include("stata_margins.jl")
include("stata_estat.jl")
include("stata_boxcox.jl")
include("stata_sureg.jl")
include("stata_survey.jl")
include("stata_regress_cluster.jl")
include("stata_ivregress.jl")
include("stata_iv_diagnostics.jl")
include("stata_correlate.jl")
include("stata_reg3.jl")
include("stata_treatreg.jl")
include("stata_qreg.jl")
include("stata_qcount.jl")
include("stata_xtpanel_desc.jl")
include("stata_xtreg.jl")
include("stata_xtgls.jl")
# ch09 — panel-data extensions
include("stata_xtivreg_fe.jl")
include("stata_xtivreg.jl")
include("stata_hausman_taylor.jl")
include("stata_arellano_bond.jl")
include("stata_xtdpdsys.jl")
include("stata_estat_abond_artest.jl")
include("stata_estat_abond.jl")
include("stata_xtmixed.jl")
include("stata_gllamm.jl")
include("stata_recovariance.jl")
# ch10 — nonlinear regression
include("stata_robust_se_glm.jl")
include("stata_poisson.jl")
include("stata_glm.jl")
include("stata_gmm_poisson.jl")
include("stata_nlcom.jl")
include("stata_lincom.jl")
include("stata_estat_ic.jl")
include("stata_margins_at.jl")
include("stata_margins_dydx.jl")
include("stata_mfx.jl")
# ch12 — testing methods
include("stata_nbreg.jl")
include("stata_lrtest.jl")
include("stata_testnl.jl")
include("stata_power.jl")
# ch13 — bootstrap methods
include("stata_probit.jl")
include("stata_bsample.jl")
# ch16 — tobit and selection models
include("stata_tobit.jl")
include("stata_margins_tobit.jl")
include("stata_heckman.jl")
include("stata_heckman_twostep.jl")
include("stata_sktest.jl")
include("stata_predict_heckman.jl")
include("stata_predict_tobit_lognormal.jl")
include("stata_ols_fit.jl")
# ch14 — binary outcome models
include("stata_logit.jl")
include("stata_hetprob.jl")
include("stata_ivprobit.jl")
include("stata_ivprobit_twostep.jl")
include("stata_estat_classification.jl")
include("stata_estat_gof.jl")
include("stata_estat_overid_2sls.jl")
# ch15 — multinomial & multivariate discrete-choice models
include("stata_mlogit.jl")   # primary: houses _c15_raw / _c15_erf / _c15_optimize
include("stata_mprobit.jl")
include("stata_ologit.jl")
include("stata_biprobit.jl")
include("stata_nlogit.jl")
include("stata_asclogit.jl")
include("stata_asmprobit.jl")
include("stata_mixlogit.jl")
include("stata_estat_covariance.jl")
include("stata_nlsur.jl")
include("stata_estimates_table_compare.jl")
include("stata_encode.jl")
include("stata_label_list.jl")
include("stata_list.jl")
include("stata_table.jl")
include("stata_tabulate_two_way.jl")
# ch17 — count-data models
include("stata_ztnb.jl")     # primary: defines _c17_loggamma / _c17_rawval
include("stata_margins_dydx_ztnb.jl")
include("stata_gnbreg.jl")
include("stata_fmm_poisson.jl")
include("stata_fmm_nbreg.jl")
include("stata_estat_lcmean.jl")
include("stata_zinb.jl")
include("stata_poisson_gmm.jl")
include("stata_estat_overid_gmm.jl")
include("stata_nl_exp.jl")
include("stata_boot_cf_poisson.jl")
include("stata_estimates_stats.jl")
# ch18 — nonlinear panel-data models
include("stata_xtlogit_pa.jl")   # primary: defines _c18_loggamma / _c18_optimize
include("stata_xtlogit_re.jl")
include("stata_xtlogit_fe.jl")
include("stata_xtpoisson_pa.jl")
include("stata_xtpoisson_re.jl")
include("stata_xtpoisson_re_normal.jl")
include("stata_xtnbreg_pa.jl")
include("stata_xtnbreg_re.jl")
include("stata_xtnbreg_fe.jl")
include("stata_xttobit_re.jl")
include("stata_xtmelogit.jl")
include("stata_estimates_table_se.jl")

export stata_regress, stata_regress_cluster,
       stata_tabulate, stata_summarize, ereturn_list,
       psmatch2, teffects_ipw, kdensity,
       stata_ttest, stata_test, stata_margins,
       stata_mean, stata_svy_mean, stata_svy_regress,
       stata_sureg, stata_sureg_fit, stata_test_sureg,
       stata_reg3,
       stata_ivregress_2sls, stata_ivregress_gmm,
       stata_liml, stata_jive, stata_estimates_table,
       estat_endogenous, estat_overid, estat_firststage,
       stata_correlate, stata_treatreg,
       stata_qreg, stata_sqreg, stata_qcount, stata_margins_count,
       stata_xtdescribe, stata_xtsum, stata_xttab, stata_xttrans, stata_matlist,
       stata_xtreg_pa, stata_xtreg_re, stata_xtreg_re_print,
       stata_hausman, stata_areg,
       stata_xtregar, stata_xtscc, stata_xtgls,
       estat_hettest, estat_imtest, stata_boxcox,
       # ch09 — panel-data extensions
       stata_xtivreg_fe, stata_xtivreg, stata_hausman_taylor,
       stata_arellano_bond, stata_xtdpdsys, stata_estat_abond_artest,
       stata_estat_abond, stata_xtmixed, stata_gllamm, stata_recovariance,
       # ch10 — nonlinear regression
       stata_poisson, stata_glm, stata_gmm_poisson,
       stata_nlcom, stata_lincom, stata_estat_ic,
       stata_mfx, stata_margins_dydx, stata_margins_at, stata_robust_se_glm,
       # ch12 — testing methods
       stata_nbreg, stata_lrtest, stata_testnl, stata_power,
       # ch13 — bootstrap methods
       stata_probit, stata_bsample,
       # ch16 — tobit and selection models
       stata_tobit, stata_margins_tobit, stata_heckman, stata_heckman_twostep,
       stata_sktest, stata_predict_heckman, stata_predict_tobit_lognormal, stata_ols_fit,
       # ch14 — binary outcome models
       stata_logit, stata_hetprob, stata_ivprobit, stata_ivprobit_twostep,
       stata_estat_classification, stata_estat_gof, stata_estat_overid_2sls,
       # ch15 — multinomial models
       stata_mlogit, stata_test_mlogit, stata_margins_mlogit, stata_margins_dydx_mlogit,
       stata_mprobit, stata_ologit, stata_margins_dydx_ologit,
       stata_biprobit, stata_nlogit, stata_nlogitgen, stata_nlogittree,
       stata_asclogit, stata_clogit, stata_estat_alternatives, stata_estat_mfx_asclogit,
       stata_asmprobit, stata_mixlogit,
       stata_estat_covariance, stata_estat_correlation, stata_nlsur,
       stata_estimates_table_compare,
       stata_encode, stata_label_list, stata_list, stata_table, stata_tabulate_two_way,
       # ch17 — count-data models
       stata_ztnb, stata_margins_dydx_ztnb, stata_gnbreg,
       stata_zinb, stata_fmm_poisson, stata_fmm_nbreg, stata_estat_lcmean,
       stata_poisson_gmm, stata_estat_overid_gmm, stata_nl_exp,
       stata_boot_cf_poisson, stata_estimates_stats,
       # ch18 — nonlinear panel-data models
       stata_xtlogit_pa, stata_xtlogit_re, stata_xtlogit_fe,
       stata_xtpoisson_pa, stata_xtpoisson_re, stata_xtpoisson_re_normal,
       stata_xtnbreg_pa, stata_xtnbreg_re, stata_xtnbreg_fe,
       stata_xttobit_re, stata_xtmelogit, stata_estimates_table_se

# Compile the hot paths at precompile time instead of on first use in a notebook.
@setup_workload begin
    df = DataFrames.DataFrame(y = randn(60), x = randn(60), z = randn(60),
                   g = repeat(1:6, inner = 10), t = repeat(0:1, 30))
    @compile_workload begin
        redirect_stdout(devnull) do
            stata_regress(df, y = :y, x = [:x], vce = :robust)
            stata_regress(df, y = :y, x = [:x, :z], vce = :cluster, cluster = :g)
            stata_regress(df, y = :y, x = [:x], absorb = :g, vce = :cluster, cluster = :g)
            stata_regress(df, y = :y, x = [:x, (:x, :z)], nocons = true, vce = :ols)
            stata_summarize(df, :x)
            stata_tabulate(df, :t)
            kdensity(df.x)
        end
    end
end

end # module
