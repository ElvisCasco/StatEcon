# (deps provided by the StatEcon module)

# One-way tabulation that mimics Stata's `tabulate varname`.
#
#   stata_tabulate(df, :train)
#
# Prints a Stata-style table (Freq. / Percent / Cum. with a Total row) and
# returns the underlying DataFrame so the result can be reused downstream.
#
# The left-column header reproduces Stata's behaviour of wrapping the variable
# label (when present, e.g. from a .dta file read with ReadStatTables) across
# several lines; if no label is stored it falls back to the column name.

# Insert thousands separators, Stata-style: 1130 -> "1,130".
function _comma(n::Integer)
    s = string(abs(n))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    (n < 0 ? "-" : "") * join(parts, ",")
end

# Greedily wrap `text` into lines no wider than `width`.
function _wrap(text::AbstractString, width::Int)
    lines = String[]
    line = ""
    for word in split(text)
        if isempty(line)
            line = word
        elseif length(line) + 1 + length(word) <= width
            line *= " " * word
        else
            push!(lines, line)
            line = word
        end
    end
    isempty(line) || push!(lines, line)
    isempty(lines) ? [""] : lines
end

# Retrieve the stored variable label for `col`, if any.
function _var_label(df::AbstractDataFrame, col::Symbol)
    try
        lbl = colmetadata(df, col, "label", "")
        return isempty(String(lbl)) ? string(col) : String(lbl)
    catch
        return string(col)
    end
end

function stata_tabulate(df::AbstractDataFrame, col::Symbol;
                        missingok::Bool=false, label::Union{Nothing,AbstractString}=nothing)
    v = df[!, col]
    missingok || (v = collect(skipmissing(v)))

    # Frequencies by value, sorted ascending (Stata's default ordering).
    counts = Dict{Any,Int}()
    for x in v
        counts[x] = get(counts, x, 0) + 1
    end
    keys_sorted = sort(collect(keys(counts)); lt = (a, b) -> isless(a, b))
    total = sum(values(counts))

    valstrs  = [string(k) for k in keys_sorted]
    freqs    = [counts[k] for k in keys_sorted]
    freqstrs = [_comma(f) for f in freqs]
    pcts     = [100f / total for f in freqs]
    cums     = cumsum(pcts)

    # ---- column widths -------------------------------------------------
    label   = isnothing(label) ? _var_label(df, col) : String(label)
    valwidth = maximum(length.(vcat(valstrs, "Total")))
    labwidth = maximum(length.(_wrap(label, max(valwidth, 8))))
    lw = max(valwidth, labwidth, 5)                 # left column width

    freqw = max(maximum(length.(vcat(freqstrs, _comma(total)))), 5)
    pctw, cumw = 7, 7                               # e.g. "100.00"

    pad(s, w)  = lpad(s, w)
    sep()      = println("-"^lw * "-+-" * "-"^(freqw + pctw + cumw + 6))

    # ---- header --------------------------------------------------------
    labellines = _wrap(label, lw)
    for (i, ln) in enumerate(labellines)
        if i < length(labellines)
            println(pad(ln, lw) * " |")
        else
            println(pad(ln, lw) * " | " *
                    pad("Freq.", freqw) * "   " *
                    pad("Percent", pctw) * "   " *
                    pad("Cum.", cumw))
        end
    end
    sep()

    # ---- body ----------------------------------------------------------
    for i in eachindex(valstrs)
        println(pad(valstrs[i], lw) * " | " *
                pad(freqstrs[i], freqw) * "   " *
                pad(Printf.@sprintf("%.2f", pcts[i]), pctw) * "   " *
                pad(Printf.@sprintf("%.2f", cums[i]), cumw))
    end
    sep()

    # ---- total ---------------------------------------------------------
    println(pad("Total", lw) * " | " *
            pad(_comma(total), freqw) * "   " *
            pad(Printf.@sprintf("%.2f", 100.0), pctw) * "   " *
            pad("", cumw))

    return DataFrames.DataFrame(Value = valstrs, Freq = freqs, Percent = pcts, Cum = cums)
end
