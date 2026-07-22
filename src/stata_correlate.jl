# --------------------------------------------------------------------------
# stata_correlate — Stata-style `correlate varlist [if …]`.
# Prints the lower-triangular Pearson correlation matrix with 4-decimal
# display and Stata's 8-char variable-name truncation via `~`.
# --------------------------------------------------------------------------

"""
    stata_correlate(df, vars; filter_missing=Symbol[])

Stata-style `correlate varlist`. Drops rows with any missing value on
`vars ∪ filter_missing`, computes the Pearson correlation on the survivors,
and prints the lower triangle with 4-decimal cells. Variable names longer
than 8 chars are truncated with `~` (Stata convention).

Returns the full `n × n` correlation matrix.
"""
function stata_correlate(df, vars::AbstractVector; filter_missing = Symbol[])
    vars_v   = [Symbol(v) for v in vars]
    filter_v = [Symbol(v) for v in filter_missing]
    d        = DataFrames.dropmissing(df, unique(vcat(vars_v, filter_v)))
    M        = hcat([Float64.(_sm_rawval.(d[!, v])) for v in vars_v]...)
    R        = Statistics.cor(M)
    n_obs    = DataFrames.nrow(d)
    n_var    = length(vars_v)

    _trunc(s) = length(s) <= 8 ? s : s[1:6] * "~" * s[end:end]
    nms      = string.(vars_v)
    nms_disp = _trunc.(nms)

    Printf.@printf("(obs = %d)\n", n_obs)
    hdr = Printf.@sprintf("%12s |", "")
    for nm in nms_disp
        hdr *= Printf.@sprintf(" %8s", nm)
    end
    println(hdr)
    println("-"^13 * "+" * "-"^(9 * n_var))

    for i in 1:n_var
        row = Printf.@sprintf("%12s |", nms[i])
        for j in 1:i
            row *= Printf.@sprintf("%9.4f", R[i, j])
        end
        println(row)
    end
    return R
end
