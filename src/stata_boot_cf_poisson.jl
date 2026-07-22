# ============================================================================
# stata_boot_cf_poisson.jl — bootstrap control-function Poisson (C&T ch17)
#
# Command-private helpers (prefix `_c17_`) live here: `_c17_obs_bootstrap`
# (observation resampling) and `_c17_boot_table_stata` (Stata `Bootstrap
# results` printer). They are the src analogues of the qmd's pedagogical
# `obs_bootstrap` / `boot_table_stata` demos.
# ============================================================================

"""
    _c17_obs_bootstrap(f, df; B=400, seed=10101, verbose=true)

Observation-level bootstrap — resamples n rows with replacement (no cluster
structure). Matches Stata's `vce(bootstrap)` when no cluster is specified.
Returns a `B × k` matrix of the replicated statistics `f(resample)`.
"""
function _c17_obs_bootstrap(f::Function, df; B::Int=400, seed::Int=10101,
                            verbose::Bool=true)
    rng = Random.MersenneTwister(seed)
    n = DataFrames.nrow(df)
    results = Vector{Vector{Float64}}()
    n_fail = 0
    for b in 1:B
        row_idx = StatsBase.sample(rng, 1:n, n; replace=true)
        sub = df[row_idx, :]
        try
            push!(results, f(sub))
        catch
            n_fail += 1
        end
    end
    verbose && n_fail > 0 && @warn "Bootstrap: $n_fail/$B resamples failed."
    isempty(results) && error("All bootstrap resamples failed.")
    return reduce(hcat, results)'
end

"""
    _c17_boot_table_stata(β, β_boot; coefnames, n, level=0.95)

Stata's `Bootstrap results` block. `β` is the observed (full-sample) point
estimate, `β_boot` a `B × k` matrix of replicates; `coefnames`/`β` are in
display order (slopes first, `_cons` last). Bootstrap SE = sd of the
replicates; CIs are the normal-based β ± z·SE. Returns `(; se, z, p)`.
"""
function _c17_boot_table_stata(β::AbstractVector, β_boot::AbstractMatrix;
                               coefnames, n::Integer, level::Float64 = 0.95)
    se   = vec(Statistics.std(β_boot, dims = 1))
    z    = β ./ se
    p    = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    B    = size(β_boot, 1)

    function g9(x; w::Int = 10, sig::Int = 7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        su = sig; s = Printf.@sprintf("%.*g", su, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && su > 1
            su -= 1; s = Printf.@sprintf("%.*g", su, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w)
    end
    commafmt(num::Integer) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    nstr = commafmt(n); bstr = commafmt(B)
    vw   = max(length(nstr), length(bstr))      # value field = widest value
    println()
    Printf.@printf("%-57s%-13s = %*s\n", "Bootstrap results",
                   "Number of obs", vw, nstr)
    Printf.@printf("%57s%-13s = %*s\n", "", "Replications", vw, bstr)
    println()
    println("-"^78)
    println("             |   Observed   Bootstrap                         Normal-based")
    Printf.@printf("%12s | coefficient  std. err.      z    P>|z|     [%g%% conf. interval]\n",
                   "", 100*level)
    println("-"^13, "+", "-"^64)
    for i in eachindex(β)
        lo = β[i] - crit*se[i]; hi = β[i] + crit*se[i]
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       string(coefnames[i]), g9(β[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%7.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       g9(lo; w=9), g9(hi; w=10))
    end
    println("-"^78)
    return (; se, z, p)
end

"""
    stata_boot_cf_poisson(df; count_dep, endog, exog, instruments,
                          B=400, seed=10101, level=0.95, title="",
                          resid_label="lpuhat2") -> NamedTuple

Bootstrap of the control-function two-step Poisson estimator — the Julia
counterpart of Stata's `program endogtwostep … ; bootstrap _b, reps()
seed()`. Each replicate (and the point estimate):

  1. OLS first stage: `endog ~ exog + instruments`; residual = control fn.
  2. Poisson second stage: `count_dep ~ endog + exog + residual`.

Resampling the whole two-step propagates first-stage estimation
uncertainty into the SEs (the naive robust SEs on the single second-stage
fit omit it). Uses `_c17_obs_bootstrap` (observation resampling) and prints
via `_c17_boot_table_stata`. Returns `(; β, β_boot, coefnames)`.
"""
function stata_boot_cf_poisson(df; count_dep::Symbol, endog::Symbol,
                               exog::AbstractVector{Symbol},
                               instruments::AbstractVector{Symbol},
                               B::Int = 400, seed::Int = 10101,
                               level::Float64 = 0.95, title::String = "",
                               resid_label::AbstractString = "lpuhat2")
    cols = unique(vcat([count_dep, endog], collect(exog), collect(instruments)))
    dfc  = DataFrames.dropmissing(df[:, cols])
    resid_sym = :_cf_resid_
    fml1 = StatsModels.term(endog) ~
           sum(StatsModels.term.(vcat(collect(exog), collect(instruments))))
    fml2 = StatsModels.term(count_dep) ~
           sum(StatsModels.term.(vcat([endog], collect(exog), [resid_sym])))

    function _step(sub)
        m1 = FixedEffectModels.reg(sub, fml1)
        s2 = DataFrames.copy(sub)
        s2[!, resid_sym] = s2[!, endog] .- FixedEffectModels.predict(m1, s2)
        m2 = GLM.glm(fml2, s2, Distributions.Poisson(), GLM.LogLink())
        return GLM.coef(m2)
    end

    # Point estimate + coefnames from the full sample.
    s0 = DataFrames.copy(dfc)
    m1_0 = FixedEffectModels.reg(s0, fml1)
    s0[!, resid_sym] = s0[!, endog] .- FixedEffectModels.predict(m1_0, s0)
    m2_0 = GLM.glm(fml2, s0, Distributions.Poisson(), GLM.LogLink())
    β  = GLM.coef(m2_0)
    cn_raw = GLM.coefnames(m2_0)
    β_boot = _c17_obs_bootstrap(_step, dfc; B = B, seed = seed)

    # Reorder to Stata display order: slopes (formula order) first,
    # `_cons` last; relabel the control-function residual.
    cons_idx = findfirst(==("(Intercept)"), cn_raw)
    slope_idx = setdiff(1:length(β), cons_idx === nothing ? Int[] : [cons_idx])
    ord = vcat(slope_idx, cons_idx === nothing ? Int[] : [cons_idx])
    relabel(c) = c == "(Intercept)" ? "_cons" :
                 c == string(resid_sym) ? resid_label : c
    cn = [relabel(cn_raw[i]) for i in ord]

    isempty(title) || println(title)
    _c17_boot_table_stata(β[ord], β_boot[:, ord];
                     coefnames = cn, n = DataFrames.nrow(dfc), level = level)
    return (; β = β[ord], β_boot = β_boot[:, ord], coefnames = cn)
end
