"""
    stata_tabulate_two_way(df, rowvar, colvar; keep_missing=false)

Two-way frequency table, à la Stata `tabulate <rowvar> <colvar>`.
Rows = unique values of `rowvar`, columns = unique values of `colvar`,
sorted by the underlying raw value (so labeled categoricals appear in
key order). Labels are unwrapped to their string form. Missing values
are dropped unless `keep_missing = true`. Prints the cross-tabulation
with row / column / grand totals.
"""
function stata_tabulate_two_way(df::DataFrames.AbstractDataFrame,
                                rowvar::Symbol, colvar::Symbol;
                                keep_missing::Bool = false)
    rcol = df[!, rowvar]; ccol = df[!, colvar]
    mask = keep_missing ? trues(length(rcol)) :
                          .!ismissing.(rcol) .& .!ismissing.(ccol)
    r_raw = [_c15_raw(v) for v in rcol[mask]]
    c_raw = [_c15_raw(v) for v in ccol[mask]]

    function _label(v)
        ismissing(v) && return "."
        if hasproperty(v, :labels) && hasproperty(v, :value)
            return haskey(v.labels, v.value) ? v.labels[v.value] : string(v.value)
        end
        return string(v)
    end
    r_lab = Dict{Any,String}()
    for (raw, v) in zip(r_raw, rcol[mask])
        haskey(r_lab, raw) || (r_lab[raw] = _label(v))
    end
    c_lab = Dict{Any,String}()
    for (raw, v) in zip(c_raw, ccol[mask])
        haskey(c_lab, raw) || (c_lab[raw] = _label(v))
    end

    r_keys = sort(unique(r_raw))
    c_keys = sort(unique(c_raw))
    counts = Dict{Tuple{Any,Any}, Int}()
    for (rk, ck) in zip(r_raw, c_raw)
        counts[(rk, ck)] = get(counts, (rk, ck), 0) + 1
    end
    row_tot = Dict(rk => sum(get(counts, (rk, ck), 0) for ck in c_keys)
                   for rk in r_keys)
    col_tot = Dict(ck => sum(get(counts, (rk, ck), 0) for rk in r_keys)
                   for ck in c_keys)
    grand   = sum(values(row_tot))

    _comma(n::Integer) = replace(string(n), r"(\d)(?=(\d{3})+$)" => s"\1,")
    row_lbls = [r_lab[k] for k in r_keys]
    col_lbls = [c_lab[k] for k in c_keys]
    row_w = max(length(string(rowvar)), maximum(length, row_lbls; init = 0),
                length("Total"))
    cell_w = max(maximum(length ∘ _comma ∘ Int,
                         values(counts); init = 1),
                 maximum(length, col_lbls; init = 0),
                 length("Total")) + 2

    # Top header line: blank then `colvar` centred-ish
    print(lpad("", row_w), " | ", string(colvar))
    println()
    # Column-name row
    print(lpad(string(rowvar), row_w), " |")
    for cl in col_lbls; print(lpad(cl, cell_w)); end
    Printf.@printf("%*s\n", cell_w, "Total")
    println("-"^(row_w + 1), "+", "-"^(cell_w * (length(c_keys) + 1)))
    for (i, rk) in enumerate(r_keys)
        print(lpad(row_lbls[i], row_w), " |")
        for ck in c_keys
            print(lpad(_comma(get(counts, (rk, ck), 0)), cell_w))
        end
        Printf.@printf("%*s\n", cell_w, _comma(row_tot[rk]))
    end
    println("-"^(row_w + 1), "+", "-"^(cell_w * (length(c_keys) + 1)))
    print(lpad("Total", row_w), " |")
    for ck in c_keys; print(lpad(_comma(col_tot[ck]), cell_w)); end
    Printf.@printf("%*s\n", cell_w, _comma(grand))
end
