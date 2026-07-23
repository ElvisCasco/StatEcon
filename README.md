# StatEcon

Julia equivalents of common **Stata** commands, printing output in Stata's own format.

Written to make Stata-to-Julia translations readable side by side: the numbers come from
Julia's econometrics stack (`FixedEffectModels`, `GLM`, `Optim`), but the tables look like
Stata's. The package covers the estimators used throughout Cameron & Trivedi,
*Microeconometrics Using Stata* — **106 commands** across linear, nonlinear,
limited-dependent-variable, discrete-choice and panel models — and ships the book's
datasets so every example runs out of the box.

## Installation

```julia
import Pkg
Pkg.add(url = "https://github.com/ElvisCasco/StatEcon")
```

Requires Julia 1.10 or newer.

## Quick start

```julia
using StatEcon

auto = dataset("auto")                                    # bundled, loaded by name
stata_regress(auto, y = :mpg, x = [:weight, :length], vce = :robust)
```

```
Linear regression                               Number of obs =     74
                                                F(2, 71)      =  55.38
                                                Prob > F      = 0.0000
                                                R-squared     = 0.6614
                                                Root MSE      = 3.4137
-----------------------------------------------------------------------
         |                  Robust
     mpg | Coefficient  std. err.      t  P>|t|   [95% conf.  interval]
---------+-------------------------------------------------------------
  weight |   -.0038515   .0019879  -1.94  0.057    -.0078152   .0001122
  length |   -.0795935   .0691697  -1.15  0.254     -.217514   .0583271
   _cons |  47.8848733  7.6629582   6.25  0.000   32.6053686  63.164378
-----------------------------------------------------------------------
```

Estimation commands return the fitted object, so the usual accessors keep working:

```julia
using FixedEffectModels          # StatEcon uses it internally but does not re-export it

m = stata_regress(auto, y = :mpg, x = [:weight], vce = :robust);
auto.uh = FixedEffectModels.residuals(m, auto)
```

## Data

The 31 datasets used throughout the book ship with the package and are loaded **by name** —
no paths, no folder layout to remember:

```julia
auto = dataset("auto")          # data/auto.dta            -> DataFrame
df10 = dataset("mus10data")     # data/musr/mus10data.dta  -> DataFrame

datasets()                      # every bundled file
```

The name may be a bare stem or a full file name. When a stem exists in several formats the
`.dta` version wins, so ask for `dataset("mus14gdata.csv")` when you want the CSV. `.dta`
files are read with `ReadStatTables`, `.csv`/`.txt` with the `DelimitedFiles` stdlib —
`delim` and `header` are forwarded:

```julia
psid  = dataset("mus02psid92m.txt"; delim = '^', header = [:er30001, :er30002, ...])
file2 = dataset("mus202file2.csv"; header = [:name, :age, :female, :income])
```

For a file that needs custom parsing (e.g. fixed-width), ask for the path instead:

```julia
datapath("mus202file3.txt")     # absolute path to one bundled file
datadir()                       # the bundled data directory (read-only once installed)
```

## Commands

Each command lives in its own `src/<command>.jl` file, takes a `DataFrame`, and prints a
Stata-format table.

| Stata | `StatEcon` |
| --- | --- |
| `regress y x, vce(robust)` | `stata_regress(df, y = :y, x = [:x], vce = :robust)` |
| `xtreg y x, fe vce(cluster id)` | `stata_regress(df, y = :y, x = [:x], absorb = :id, vce = :cluster, cluster = :id)` |
| `summarize var [if] [, detail]` | `stata_summarize(df, :var)` |
| `tabulate var` | `stata_tabulate(df, :var)` |
| `poisson y x, vce(robust)` | `stata_poisson(df, @formula(y ~ x); vce = :robust)` |
| `probit y x` / `logit y x` | `stata_probit(df, @formula(y ~ x))` / `stata_logit(...)` |
| `tobit y x, ll(0)` | `stata_tobit(df, :y, [:x]; ll = 0.0)` |
| `ivregress 2sls y (x = z) w` | `stata_ivregress_2sls(df, :y, :x => :z, [:w])` |
| `xtreg y x, re` | `stata_xtreg_re(df, :y, [:x], :id)` |
| `margins, dydx(*)` | `stata_margins_dydx(β, V, X; ...)` |

### Coverage

* **Linear** — `stata_regress`, `stata_regress_cluster`, `stata_areg`, `stata_sureg`,
  `stata_reg3`, `stata_qreg`, `stata_sqreg`, `stata_boxcox`
* **Instrumental variables** — `stata_ivregress_2sls`, `stata_ivregress_gmm`, `stata_liml`,
  `stata_jive`, `estat_endogenous`, `estat_overid`, `estat_firststage`
* **Count / GLM** — `stata_poisson`, `stata_glm`, `stata_nbreg`, `stata_gnbreg`,
  `stata_ztnb`, `stata_zinb`, `stata_fmm_poisson`, `stata_fmm_nbreg`, `stata_poisson_gmm`,
  `stata_qcount`
* **Binary / limited dependent** — `stata_probit`, `stata_logit`, `stata_hetprob`,
  `stata_ivprobit`, `stata_ivprobit_twostep`, `stata_tobit`, `stata_heckman`,
  `stata_heckman_twostep`, `stata_treatreg`
* **Multinomial / discrete choice** — `stata_mlogit`, `stata_mprobit`, `stata_ologit`,
  `stata_biprobit`, `stata_nlogit`, `stata_asclogit`, `stata_clogit`, `stata_asmprobit`,
  `stata_mixlogit`, `stata_nlsur`
* **Linear panel** — `stata_xtreg_re`, `stata_xtreg_pa`, `stata_xtregar`, `stata_xtgls`,
  `stata_xtscc`, `stata_xtivreg`, `stata_xtivreg_fe`, `stata_hausman_taylor`,
  `stata_arellano_bond`, `stata_xtdpdsys`, `stata_xtmixed`, `stata_gllamm`
* **Nonlinear panel** — `stata_xtlogit_pa|re|fe`, `stata_xtpoisson_pa|re`,
  `stata_xtnbreg_pa|re|fe`, `stata_xttobit_re`, `stata_xtmelogit`
* **Panel description** — `stata_xtdescribe`, `stata_xtsum`, `stata_xttab`, `stata_xttrans`
* **Testing / post-estimation** — `stata_test`, `stata_lrtest`, `stata_testnl`,
  `stata_nlcom`, `stata_lincom`, `stata_hausman`, `stata_margins`, `stata_margins_dydx`,
  `stata_mfx`, `stata_estimates_table`, `ereturn_list`, `estat_hettest`, `estat_imtest`,
  `stata_estat_ic`, `stata_estat_classification`, `stata_estat_gof`
* **Treatment effects / misc** — `psmatch2`, `teffects_ipw`, `kdensity`, `stata_ttest`,
  `stata_correlate`, `stata_bsample`, `stata_sktest`

For the full list: `names(StatEcon)`.

## Examples

`examples/Cameron_Trivedi/` reproduces **all 18 chapters** of *Microeconometrics Using
Stata* as Quarto notebooks (`ch01 … ch18`). Each one keeps the book's original Stata
commands in `#= ... =#` comment blocks next to the Julia that reproduces them, so the two
can be read side by side, and calls `StatEcon` rather than defining estimators inline.

The notebooks have their own environment, so rendering works out of the box:

```bash
cd examples/Cameron_Trivedi
quarto render ch01_Stata_basics.qmd
```

Each renders to a self-contained HTML file (no side-car `_files/` directory).

| | |
| --- | --- |
| ch01–ch04 | Stata basics, data management, linear regression, simulation |
| ch05–ch07 | GLS, instrumental variables, quantile regression |
| ch08–ch09 | Linear panel data — basics and extensions |
| ch10–ch13 | Nonlinear regression, optimization, testing, bootstrap |
| ch14–ch18 | Binary, multinomial, tobit/selection, count, nonlinear panel |

## Source of the Stata code and data

The Stata commands reproduced in the examples, and the datasets bundled under `data/`,
come from the book and its companion material:

> Cameron, A. Colin, and Pravin K. Trivedi. *Microeconometrics Using Stata*.
> College Station, TX: Stata Press.
> (Revised edition, 2010; second edition, 2022.)

* Datasets are the book's companion files distributed by **Stata Press**
  (<https://www.stata-press.com/data/>). The `data/musr/` folder holds the revised-edition
  files (`mus03data.dta`, `mus10data.dta`, …) and `data/mus2/` the second-edition ones;
  the names match those used in the book's own do-files.
* `auto.dta` and `census.dta` are the standard Stata example datasets
  (<https://www.stata-press.com/data/r10/>); ch02 downloads `census.dta` on demand and
  caches it locally.
* The Stata code shown in the `#= ... =#` blocks of each notebook is the book's, quoted
  verbatim for comparison; the Julia translation alongside it is this package's.

Copyright in the book and its datasets remains with the authors and Stata Press. They are
included here for study and reproduction of the book's results; refer to Stata Press for
the terms that apply to redistribution.

Further reading used while porting:

* [Microeconometrics with R](https://ycroissant.github.io/micsr_book/) — a parallel R treatment
* [UCLA OARC Stata modules](https://stats.oarc.ucla.edu/stata/modules/)

## Notes and limitations

* Output is formatted to match Stata's tables; small last-digit differences from Stata are
  expected where the underlying optimizer or degrees-of-freedom convention differs.
* A few advanced estimators are faithfully simplified, matching the scope the book itself
  uses, and say so in their docstrings: `stata_xtmixed`/`stata_gllamm` handle a single
  grouping factor (crossed/nested effects are MixedModels.jl territory), `stata_asmprobit`
  uses the independent-errors variant, `stata_mixlogit` one normal random coefficient, and
  `stata_nlsur` the two-equation normal case.
* Standard errors for `psmatch2` use its own fixed-weights formula (Abadie & Imbens 2006
  show this is not fully consistent); `teffects_ipw` uses the stacked-equations sandwich,
  matching Stata's "Robust std. err.".
* `stata_regress` also accepts a formula (`stata_regress(df, @formula(y ~ x))`) or an
  already-fitted model (`stata_regress(model)`).
* The bundled `data/` directory is read-only once the package is installed; notebooks write
  any output they produce to the working directory.

## License

MIT (this package). See above regarding the book's datasets and Stata code.
