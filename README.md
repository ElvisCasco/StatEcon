# StatEcon

Julia equivalents of common **Stata** commands, printing output in Stata's own format.

Written to make Stata-to-Julia translations readable side by side: the numbers come from
Julia's econometrics stack (`FixedEffectModels`, `GLM`), but the tables look like Stata's.

## Installation

```julia
import Pkg
Pkg.add(url = "https://github.com/ElvisCasco/StatEcon")
```

## Commands

| Stata | `StatEcon` |
| --- | --- |
| `regress y x, vce(robust)` | `stata_regress(df, y = :y, x = [:x], vce = :robust)` |
| `xtreg y x, fe vce(cluster id)` | `stata_regress(df, y = :y, x = [:x], absorb = :id, vce = :cluster, cluster = :id)` |
| `tabulate var` | `stata_tabulate(df, :var)` |
| `summarize var if cond` | `stata_summarize(df, :var; if_ = cond)` |
| `psmatch2 t x, outcome(y) logit ate` | `psmatch2(df, :t, [:x]; outcome = :y)` |
| `teffects ipw (y) (t x)` | `teffects_ipw(df, :t, [:x]; outcome = :y)` |
| `kdensity var` | `kdensity(df.var)` |

Beyond the linear-model core above, `StatEcon` now covers the nonlinear and
panel toolkit used throughout Cameron & Trivedi, *Microeconometrics Using Stata*:

* **Count / GLM** — `stata_poisson`, `stata_glm`, `stata_nbreg`, `stata_gnbreg`, `stata_ztnb`, `stata_zinb`, `stata_fmm_poisson`, `stata_fmm_nbreg`, `stata_poisson_gmm`
* **Binary / limited dependent** — `stata_probit`, `stata_logit`, `stata_hetprob`, `stata_ivprobit`, `stata_tobit`, `stata_heckman`
* **Multinomial / discrete choice** — `stata_mlogit`, `stata_mprobit`, `stata_ologit`, `stata_biprobit`, `stata_nlogit`, `stata_asclogit`, `stata_clogit`, `stata_asmprobit`, `stata_mixlogit`
* **Panel** — `stata_xtreg_*`, `stata_xtivreg`, `stata_hausman_taylor`, `stata_arellano_bond`, `stata_xtdpdsys`, `stata_xtmixed`, `stata_xtlogit_*`, `stata_xtpoisson_*`, `stata_xtnbreg_*`, `stata_xttobit_re`
* **Testing / post-estimation** — `stata_test`, `stata_lrtest`, `stata_testnl`, `stata_nlcom`, `stata_lincom`, `stata_margins_*`, `stata_mfx`, `stata_estat_*`

Each command lives in its own `src/<command>.jl` file and prints Stata-format output.

## Examples

`examples/Cameron_Trivedi/` reproduces **all 18 chapters** of Cameron & Trivedi,
*Microeconometrics Using Stata*, as Quarto notebooks (`ch01…ch18`) that call
`StatEcon` instead of defining estimators inline. Data resolves automatically
whether the notebook is rendered with Quarto or run interactively.

## Example

```julia
using StatEcon, DataFrames, ReadStatTables

df = DataFrame(ReadStatTables.readstat("wagepan_did.dta"))

StatEcon.stata_regress(df, y = :D_lwage, x = [:train], vce = :robust)
```

```
Linear regression                               Number of obs =    545
                                                F(1, 543)     =   4.09
                                                Prob > F      = 0.0435
                                                R-squared     = 0.0085
                                                Root MSE      = .45454
----------------------------------------------------------------------
         |                  Robust
 D.lwage | Coefficient  std. err.     t  P>|t|   [95% conf.  interval]
---------+------------------------------------------------------------
   train |    .0960654     .04748  2.02  0.044     .0027985   .1893324
   _cons |    .0599069   .0217649  2.75  0.006     .0171532   .1026605
----------------------------------------------------------------------
```

## `stata_regress`

```julia
stata_regress(df; y, x, vce = :ols, cluster = nothing, absorb = nothing,
              nocons = false, yname = nothing, xnames = nothing, title = nothing)
```

* `y` — dependent variable (`Symbol`).
* `x` — regressors: `Symbol`s, or tuples for interactions, e.g. `(:train, :educ)` → `c.train#c.educ`.
* `vce` — `:ols`, `:robust`, or `:cluster` (with `cluster = :id`).
* `absorb` — fixed effect(s) to absorb, i.e. Stata's `xtreg ..., fe`.
* `nocons` — Stata's `nocons`.
* `yname` / `xnames` — override displayed names; `xnames` accepts a vector or `"coef" => "label"` pairs.

Naming follows Stata automatically: `D_lwage` prints as `D.lwage`, `L_uh` as `L.uh`, and
interactions as `c.a#c.b`. The intercept is shown as `_cons`, last.

It returns the fitted model, so `residuals`, `predict` and `coef` still work:

```julia
ols = StatEcon.stata_regress(df, y = :y, x = [:x], vce = :robust);
df.uh = FixedEffectModels.residuals(ols, df)
```

## Notes

* Standard errors for `psmatch2` use its own fixed-weights formula (Abadie & Imbens 2006
  show this is not fully consistent); `teffects_ipw` uses the stacked-equations sandwich,
  matching Stata's "Robust std. err.".
* `stata_regress` also accepts a formula (`stata_regress(df, @formula(y ~ x))`) or an
  already-fitted model (`stata_regress(model)`).

## License

MIT
