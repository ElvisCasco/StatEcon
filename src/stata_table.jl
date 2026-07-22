"""
    stata_table(df, byvar, stats; nototals=false, nformat=nothing)

Stata 17+ `table <byvar>, statistic(<stat> <var>) ...` for a one-way
table of summary statistics. `stats` is a vector of `(stat, var)`
symbol-tuples where `stat ∈ {:count, :mean, :sd, :min, :max, :sum,
:median}`. Labeled `byvar` values display by their string label and
sort by the underlying raw key. A Total row is included by default.
`nformat` accepts a Stata-style format string (e.g. `"%6.0f"`,
`"%9.2f"`) and overrides the default `%9.0g` for non-count columns.
"""
function stata_table(df::DataFrames.AbstractDataFrame, byvar::Symbol,
                     stats::AbstractVector{<:Tuple{Symbol,Symbol}};
                     nototals::Bool=false,
                     nformat::Union{Nothing,AbstractString}=nothing)
    col = df[!, byvar]
    mask = .!ismissing.(col)
    bycol = col[mask]
    raws = [_c15_raw(v) for v in bycol]

    _label(v) = ismissing(v) ? "." :
                (hasproperty(v, :labels) && hasproperty(v, :value) &&
                 haskey(v.labels, v.value)) ? v.labels[v.value] : string(v)
    labs = [_label(v) for v in bycol]

    keys_sorted = sort(unique(raws))
    label_for = Dict{Any,String}()
    for (r, l) in zip(raws, labs)
        haskey(label_for, r) || (label_for[r] = l)
    end

    _statfn = Dict(:count  => x -> Float64(length(x)),
                   :mean   => Statistics.mean,
                   :sd     => Statistics.std,
                   :min    => minimum, :max => maximum,
                   :sum    => sum,
                   :median => Statistics.median)

    df_kept = df[mask, :]
    group_vals = Vector{Vector{Float64}}()
    for k in keys_sorted
        idx = findall(==(k), raws)
        sub = df_kept[idx, :]
        gv = Float64[]
        for (stat, v) in stats
            xs = Float64.(_c15_raw.(skipmissing(sub[!, v])))
            push!(gv, _statfn[stat](xs))
        end
        push!(group_vals, gv)
    end
    total_vals = Float64[]
    if !nototals
        for (stat, v) in stats
            xs = Float64.(_c15_raw.(skipmissing(df_kept[!, v])))
            push!(total_vals, _statfn[stat](xs))
        end
    end

    stat_label_map = Dict(:count => "N", :mean => "Mean", :sd => "SD",
                          :min => "Min", :max => "Max", :sum => "Sum",
                          :median => "Median")
    headers = [stat_label_map[s] * "(" * string(v) * ")" for (s, v) in stats]

    _g9(x; w::Int=9, mx::Int=7) = begin
        (ismissing(x) || !isfinite(x)) && return "."
        sig = mx
        s = Printf.@sprintf("%.*g", sig, x)
        while length(s) > w && sig > 1
            sig -= 1
            s = Printf.@sprintf("%.*g", sig, x)
        end
        0 < abs(x) < 1 ? replace(s, r"^(-?)0\." => s"\1.") : s
    end
    _comma_int(x::Integer) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")
    # Stata `nformat(...)` accepts standard C/printf specs (%6.0f, %9.2f,
    # %9.3e, etc.). We strip the leading "%" and feed it to Printf via
    # Printf.Format so the runtime spec works inside @sprintf.
    _nfmt = nformat === nothing ? nothing : Printf.Format(nformat)
    _fmt(stat, x) = begin
        stat == :count        && return _comma_int(Int(round(x)))
        _nfmt !== nothing     && return strip(Printf.format(_nfmt, x))
        _g9(x)
    end

    label_strs = [label_for[k] for k in keys_sorted]
    var_w  = max(11, length(string(byvar)),
                 maximum(length, label_strs; init = 0) + 3, length("Total") + 3)
    col_ws = [max(length(h) + 3, 12) for h in headers]
    total_w = sum(col_ws)

    # Header line
    print(lpad("", var_w), " |")
    for (i, h) in enumerate(headers); print(lpad(h, col_ws[i])); end
    println()
    println("-"^(var_w + 1), "+", "-"^total_w)

    # byvar section header row
    println(rpad(string(byvar), var_w), " |")

    for i in eachindex(keys_sorted)
        print(lpad("   " * label_strs[i], var_w), " |")
        for j in eachindex(stats)
            print(lpad(_fmt(stats[j][1], group_vals[i][j]), col_ws[j]))
        end
        println()
    end
    if !nototals
        print(lpad("   Total", var_w), " |")
        for j in eachindex(stats)
            print(lpad(_fmt(stats[j][1], total_vals[j]), col_ws[j]))
        end
        println()
    end
    println("-"^(var_w + 1 + total_w + 1))
end
