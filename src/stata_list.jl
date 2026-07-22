"""
    stata_list(df, vars; in_range=:, noobs=false)

Stata `list <varlist> in <range>, clean`. Prints rows of `df` for the
named columns, with the obs-number prefix (`1.  2.  …`) and clean
formatting (no `+----+` borders, no `|` separators). Values are
%9.0g-formatted for numerics and unwrapped to their string label for
`ReadStatTables.LabeledValue` columns. Pass `noobs = true` to drop the
obs-number prefix; `in_range = i` (Integer), `i:j` (UnitRange), or `:`
(all rows) controls the row selection.
"""
function stata_list(df::DataFrames.AbstractDataFrame,
                    vars::AbstractVector{Symbol};
                    in_range = Colon(),
                    noobs::Bool = false)
    rows = in_range === Colon()    ? collect(1:DataFrames.nrow(df)) :
           in_range isa Integer    ? [in_range] :
           collect(in_range)

    _g9(x; w::Int = 9, mx::Int = 7) = begin
        (ismissing(x) || !isfinite(x)) && return "."
        sig = mx
        s = Printf.@sprintf("%.*g", sig, x)
        while length(s) > w && sig > 1
            sig -= 1; s = Printf.@sprintf("%.*g", sig, x)
        end
        0 < abs(x) < 1 ? replace(s, r"^(-?)0\." => s"\1.") : s
    end
    function _fmt(v)
        ismissing(v) && return "."
        if hasproperty(v, :labels) && hasproperty(v, :value)
            return haskey(v.labels, v.value) ?
                   v.labels[v.value] : string(v.value)
        end
        v isa Integer && return string(v)
        v isa Real    && return _g9(Float64(v))
        return string(v)
    end

    var_strs = string.(vars)
    cells = [_fmt(df[i, v]) for i in rows, v in vars]
    col_w = [max(length(var_strs[j]),
                 maximum(length, view(cells, :, j); init = 0))
             for j in 1:length(vars)]

    obs_w = noobs ? 0 : maximum(length(string(i)) for i in rows; init = 1)

    # Header
    print(noobs ? "    " : " "^(obs_w + 4))
    for j in 1:length(vars)
        print(lpad(var_strs[j], col_w[j]), "  ")
    end
    println()

    # Rows
    for (ridx, i) in enumerate(rows)
        if noobs
            print("    ")
        else
            Printf.@printf("%*s.   ", obs_w, string(i))
        end
        for j in 1:length(vars)
            print(lpad(cells[ridx, j], col_w[j]), "  ")
        end
        println()
    end
    return nothing
end
