"""
    stata_estimates_table_compare(name_res_pairs; keep,
                                  stats=[:N,:ll,:aic,:bic],
                                  bfmt="%7.3f", stfmt="%7.0f")

Stata `estimates table <names>, keep(<vars>) stats(<stats>) b(<bfmt>)
stfmt(<stfmt>)`. Side-by-side comparison of two or more stored
estimation results (`name => res` pairs). Distinct from the older
`estimates_table_stata` in this file, which takes pre-built coef-and-
SE "spec" tuples and adds t-stat rows.

For each requested variable name in `keep`, looks up the coefficient
via `res.alt_specific` + `res.β_alt`. If a model has no coefficient
for that variable (e.g., MNL on case-specific-only regressors),
prints `.` (matching Stata). The `stats` block computes:

* `:N`   — `res.n` or `res.n_obs`
* `:ll`  — `res.ll`
* `:aic` — `-2·ll + 2·nparam`     (uses `res.nparam`)
* `:bic` — `-2·ll + nparam·log(N)`
"""
function stata_estimates_table_compare(name_res_pairs::AbstractVector;
                                       keep::AbstractVector{Symbol},
                                       stats::AbstractVector{Symbol} =
                                           [:N, :ll, :aic, :bic],
                                       bfmt::AbstractString = "%7.3f",
                                       stfmt::AbstractString = "%7.0f")
    bfmt_p  = Printf.Format(bfmt)
    stfmt_p = Printf.Format(stfmt)

    # Coefficient lookup. Returns a Float64 or nothing (→ "." cell).
    function _coef(res, v)
        if hasproperty(res, :alt_specific) && hasproperty(res, :β_alt)
            i = findfirst(==(v), collect(res.alt_specific))
            i === nothing || return Float64(res.β_alt[i])
        end
        return nothing
    end

    # Helper to extract scalar statistics from a fitted result.
    function _nparam(res)
        hasproperty(res, :nparam) && return Int(res.nparam)
        hasproperty(res, :k)      && return Int(res.k)
        return missing
    end
    function _N(res)
        hasproperty(res, :n_obs) && return Int(res.n_obs)
        hasproperty(res, :n)     && return Int(res.n)
        return missing
    end
    function _stat(res, s)
        if s === :N  ; return _N(res); end
        if s === :ll ; return Float64(res.ll); end
        k = _nparam(res); N = _N(res); ll = Float64(res.ll)
        s === :aic && return -2 * ll + 2 * k
        s === :bic && return -2 * ll + k * log(N)
        error("unknown stat $s")
    end

    names = [String(p.first) for p in name_res_pairs]
    ress  = [p.second        for p in name_res_pairs]

    coef_w = max(8, maximum(length, names; init = 0) + 2)
    label_w = max(8, maximum(length ∘ string, keep; init = 0))

    # Header
    print(rpad("Variable", label_w + 1), "|")
    for n in names; print(lpad(n, coef_w)); end
    println()
    println("-"^(label_w + 1), "+", "-"^(coef_w * length(names)))

    # Coefficient block
    for v in keep
        print(rpad("  " * string(v), label_w + 1), "|")
        for res in ress
            c = _coef(res, v)
            cell = c === nothing ? "." : strip(Printf.format(bfmt_p, c))
            print(lpad(cell, coef_w))
        end
        println()
    end
    println("-"^(label_w + 1), "+", "-"^(coef_w * length(names)))

    # Stats block
    println(rpad("Statistics", label_w + 1), "|")
    for s in stats
        print(rpad("  " * string(s), label_w + 1), "|")
        for res in ress
            v = _stat(res, s)
            cell = v === missing ? "." :
                   v isa AbstractFloat ?
                       strip(Printf.format(stfmt_p, v)) :
                       string(v)
            print(lpad(cell, coef_w))
        end
        println()
    end
    println("-"^(label_w + 1), "+", "-"^(coef_w * length(names)))
    return nothing
end
