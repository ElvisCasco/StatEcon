"""
    StatEcon

Julia equivalents of common Stata commands, with Stata-style printed output:

  - `stata_regress`   -> `regress y x, vce(robust)` / `xtreg ..., fe`
  - `stata_tabulate`  -> `tabulate var`
  - `stata_summarize` -> `summarize var [if]`
  - `psmatch2`        -> `psmatch2 treat covars, outcome(y) logit ate`
  - `teffects_ipw`    -> `teffects ipw (y) (treat covars)`
  - `kdensity`        -> `kdensity var` (Gaussian KDE, Silverman bandwidth)

Calls are normally written module-qualified, e.g. `StatEcon.stata_regress(...)`.
"""
module StatEcon

using DataFrames, ReadStatTables, GLM, FixedEffectModels, StatsModels, Statistics, Printf
using PrecompileTools

include("stata_tabulate.jl")
include("stata_summarize.jl")
include("kdensity.jl")
include("psmatch2.jl")
include("teffects_ipw.jl")
include("stata_regress.jl")

export stata_regress, stata_tabulate, stata_summarize, psmatch2, teffects_ipw, kdensity

# Compile the hot paths at precompile time instead of on first use in a notebook.
@setup_workload begin
    df = DataFrame(y = randn(60), x = randn(60), z = randn(60),
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
