# Stat_Econ

Julia equivalents of common **Stata** commands, printing output in Stata's own format.

Written to make Stata-to-Julia translations readable side by side: the numbers come from
Julia's econometrics stack (`FixedEffectModels`, `GLM`), but the tables look like Stata's.

## Installation

```julia
import Pkg
Pkg.add(url = "https://github.com/ElvisCasco/Stat_Econ")
```

## Commands

| Stata | `Stat_Econ` |
| --- | --- |
| `regress y x, vce(robust)` | `stata_regress(df, y = :y, x = [:x], vce = :robust)` |
| `xtreg y x, fe vce(cluster id)` | `stata_regress(df, y = :y, x = [:x], absorb = :id, vce = :cluster, cluster = :id)` |
| `tabulate var` | `stata_tabulate(df, :var)` |
| `summarize var if cond` | `stata_summarize(df, :var; if_ = cond)` |
| `psmatch2 t x, outcome(y) logit ate` | `psmatch2(df, :t, [:x]; outcome = :y)` |
| `teffects ipw (y) (t x)` | `teffects_ipw(df, :t, [:x]; outcome = :y)` |
| `kdensity var` | `kdensity(df.var)` |

## Example

```julia
using Stat_Econ, DataFrames, ReadStatTables

df = DataFrame(ReadStatTables.readstat("wagepan_did.dta"))

Stat_Econ.stata_regress(df, y = :D_lwage, x = [:train], vce = :robust)
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
ols = Stat_Econ.stata_regress(df, y = :y, x = [:x], vce = :robust);
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
