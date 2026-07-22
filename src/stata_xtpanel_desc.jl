# =============================================================================
# Stata-style panel descriptive commands: stata_xtdescribe / stata_xtsum / stata_xttab / stata_xttrans
# plus stata_matlist utility.  Ported from Cameron & Trivedi ch08 (module-qualified
# names, no `using` clauses here â€” that is the whole-module concern of
# `StatEcon.jl`).  PrettyTables usage in the source has been rewritten to plain
# `println(DataFrame(...))` following the ch06 cleanup convention.
# =============================================================================

"""
    stata_xtdescribe(df, idvar, timevar; max_patterns=9, xtset=true) -> NamedTuple

Stata-style `xtset` + `stata_xtdescribe`. Prints:
  - The `xtset` preamble (Panel variable / Time variable / Delta) when
    `xtset=true`. Detects (un)balanced panels and `but with gaps`.
  - The `stata_xtdescribe` block:
      • id range + n
      • time range + T (max periods)
      • Delta(year), Span(year), uniqueness statement
      • Distribution of T_i (min, 5%, 25%, 50%, 75%, 95%, max)
      • Frequency / Percent / Cum. / Pattern table — top
        `max_patterns` patterns plus an "(other patterns)" row.

Returns `(; N, T_obs, T_dist, balanced)`.
"""
function stata_xtdescribe(df, idvar::Symbol, timevar::Symbol;
                    max_patterns::Int = 9, xtset::Bool = true)
    ids_all  = df[!, idvar]
    tims_all = df[!, timevar]
    keep = .!ismissing.(ids_all) .& .!ismissing.(tims_all)
    ids  = ids_all[keep];  tims = tims_all[keep]

    unique_ids  = sort(unique(ids))
    unique_tims = sort(unique(tims))
    n_panels    = length(unique_ids)
    T_max       = length(unique_tims)
    t_min       = minimum(unique_tims)
    t_mx        = maximum(unique_tims)

    T_obs = DataFrames.combine(
        DataFrames.groupby(DataFrames.DataFrame(_id = ids, _t = tims), :_id),
        DataFrames.nrow => :T)
    t_per_id = T_obs.T
    balanced = all(==(t_per_id[1]), t_per_id)

    # Detect "with gaps": some panel has fewer obs than its time-span.
    has_gaps = false
    for sub in DataFrames.groupby(DataFrames.DataFrame(_id = ids, _t = tims), :_id)
        ts = sort(sub._t)
        if length(ts) > 1 && (ts[end] - ts[1] + 1) > length(ts)
            has_gaps = true; break
        end
    end

    # Pattern strings (length T_max, '1' = observed, '.' = missing).
    t_to_idx = Dict(t => i for (i, t) in enumerate(unique_tims))
    pat_count = Dict{String, Int}()
    for sub in DataFrames.groupby(DataFrames.DataFrame(_id = ids, _t = tims), :_id)
        pat = fill('.', T_max)
        for t in sub._t
            pat[t_to_idx[t]] = '1'
        end
        s = String(pat)
        pat_count[s] = get(pat_count, s, 0) + 1
    end
    sorted_patterns = sort(collect(pat_count); by = x -> -x[2])

    # Print the `xtset` preamble.
    if xtset
        bal_str = balanced ? "(strongly balanced)" : "(unbalanced)"
        gap_str = has_gaps ? ", but with gaps" : ""
        println("Panel variable: $idvar $bal_str")
        Printf.@printf(" Time variable: %s, %s to %s%s\n",
                       string(timevar), string(t_min), string(t_mx), gap_str)
        println("         Delta: 1 unit")
        println()
        println(". stata_xtdescribe")
        println()
    end

    # id summary line: "  id:  v1, v2, ..., vlast                       n =     <n>"
    id_str = if length(unique_ids) >= 3
        Printf.@sprintf("%s, %s, ..., %s",
                        string(unique_ids[1]), string(unique_ids[2]),
                        string(unique_ids[end]))
    else
        join(string.(unique_ids), ", ")
    end
    Printf.@printf("%9s:  %-50s n = %10d\n", string(idvar), id_str, n_panels)

    # time summary
    tim_str = if length(unique_tims) >= 3
        Printf.@sprintf("%s, %s, ..., %s",
                        string(unique_tims[1]), string(unique_tims[2]),
                        string(unique_tims[end]))
    else
        join(string.(unique_tims), ", ")
    end
    Printf.@printf("%9s:  %-50s T = %10d\n", string(timevar), tim_str, T_max)
    Printf.@printf("           Delta(%s) = 1 unit\n", string(timevar))
    Printf.@printf("           Span(%s)  = %d periods\n", string(timevar), T_max)
    Printf.@printf("           (%s*%s uniquely identifies each observation)\n",
                   string(idvar), string(timevar))
    println()

    # Distribution of T_i — quantiles via Statistics.quantile.
    sorted_t = sort(t_per_id)
    function pctl(p)
        round(Int, Statistics.quantile(Float64.(sorted_t), p))
    end
    println("Distribution of T_i:   min      5%     25%       50%       75%     95%     max")
    Printf.@printf("                       %3d     %3d     %3d       %3d       %3d     %3d     %3d\n",
                   minimum(sorted_t), pctl(0.05), pctl(0.25), pctl(0.50),
                   pctl(0.75), pctl(0.95), maximum(sorted_t))
    println()

    # Pattern table.
    println("     Freq.  Percent    Cum. |  Pattern")
    println(" ---------------------------+---------")
    cum_pct = 0.0
    n_top = min(max_patterns, length(sorted_patterns))
    for i in 1:n_top
        pat, cnt = sorted_patterns[i]
        pct = 100.0 * cnt / n_panels
        cum_pct += pct
        Printf.@printf(" %8d  %7.2f  %6.2f |  %s\n",
                       cnt, pct, cum_pct, pat)
    end
    if length(sorted_patterns) > n_top
        rem = sum(p[2] for p in sorted_patterns[n_top+1:end])
        rem_pct = 100.0 * rem / n_panels
        cum_pct += rem_pct
        Printf.@printf(" %8d  %7.2f  %6.2f | (other patterns)\n",
                       rem, rem_pct, cum_pct)
    end
    println(" ---------------------------+---------")
    Printf.@printf(" %8d  %7.2f         |  %s\n",
                   n_panels, 100.0, "X"^T_max)

    T_dist = DataFrames.sort(
        DataFrames.combine(DataFrames.groupby(T_obs, :T), DataFrames.nrow => :panels), :T)
    return (; N = n_panels, T_obs, T_dist, balanced)
end

"""
    stata_xtsum(df, vars, idvar)

Stata-style `stata_xtsum`. For each variable in `vars`, reports the overall, between
(std. of panel means), and within (std. of deviations from panel means) SDs.
"""
function stata_xtsum(df, vars::AbstractVector, idvar::Symbol)
    vs  = [Symbol(v) for v in vars]
    # Work only on rows complete on id + every requested variable.
    d = DataFrames.dropmissing(df, unique(vcat(idvar, vs)))
    n_panels = length(unique(d[!, idvar]))
    N_total  = DataFrames.nrow(d)
    T_cnt    = DataFrames.combine(DataFrames.groupby(d, idvar),
                                  DataFrames.nrow => :_T)._T
    balanced = all(==(T_cnt[1]), T_cnt)
    T_lbl    = balanced ? "T"   : "T-bar"
    T_str    = balanced ? string(T_cnt[1]) :
               Printf.@sprintf("%g", N_total / n_panels)

    # Stata's %9.0g-style: up to 7 sig digits, strip leading 0 for |x|<1,
    # show integers without decimals.
    function _g(x)
        (ismissing(x) || !isfinite(x)) && return "."
        if isinteger(x) && abs(x) < 1e12
            return string(Int(round(x)))
        end
        s = Printf.@sprintf("%.7g", x)
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        return s
    end
    function _commafmt(num::Integer)
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    # Header
    println("Variable         |      Mean   Std. dev.       Min        Max |    Observations")
    println("-"^17, "+", "-"^44, "+", "-"^16)

    for (vi, v) in enumerate(vs)
        vals = Float64.(_sm_rawval.(d[!, v]))
        gm   = DataFrames.combine(DataFrames.groupby(d, idvar),
                  v => (x -> Statistics.mean(Float64.(_sm_rawval.(x)))) => :_mean)
        pm   = gm._mean
        dj   = DataFrames.leftjoin(d, gm, on=idvar)
        dev  = [Float64(_sm_rawval(dj[i, v])) - dj._mean[i] for i in 1:DataFrames.nrow(dj)]
        μ    = Statistics.mean(vals)

        # Right column: 16 chars total = lpad(label, 6) + " = " + lpad(value, 7).
        # Overall row (label "N", value = total obs)
        Printf.@printf("%-9s%-8s | %9s  %9s  %9s  %9s |%6s = %7s\n",
                       string(v), "overall",
                       _g(μ),
                       _g(Statistics.std(vals)),
                       _g(minimum(vals)),
                       _g(maximum(vals)),
                       "N", _commafmt(N_total))
        # Between row (blank mean; label "n", value = n_panels)
        Printf.@printf("%-9s%-8s | %9s  %9s  %9s  %9s |%6s = %7s\n",
                       "", "between",
                       "",
                       _g(Statistics.std(pm)),
                       _g(minimum(pm)),
                       _g(maximum(pm)),
                       "n", _commafmt(n_panels))
        # Within row (blank mean; min/max are deviations shifted by μ;
        # label is "T" for balanced, "T-bar" for unbalanced).
        Printf.@printf("%-9s%-8s | %9s  %9s  %9s  %9s |%6s = %7s\n",
                       "", "within ",
                       "",
                       _g(Statistics.std(dev)),
                       _g(minimum(dev) + μ),
                       _g(maximum(dev) + μ),
                       T_lbl, T_str)
        # Blank separator between variables
        if vi < length(vs)
            Printf.@printf("%-9s%-8s | %9s  %9s  %9s  %9s |\n",
                           "", "", "", "", "", "")
        end
    end
    return nothing
end

"""
    stata_xttab(df, var, idvar=nothing)

Stata-style `stata_xttab var`. With `idvar` supplied, reports three blocks matching
Stata's output exactly:

  * **Overall**  — frequency / percent across all observations.
  * **Between**  — #/% of panels that ever took that value (can sum > 100%
                   because panels that change state appear in multiple rows).
  * **Within**   — mean within-panel time share at that value, conditional
                   on the panel ever taking it.

Without `idvar`, only Overall freq/percent/cum% are shown.
"""
function stata_xttab(df, var::Symbol, idvar::Union{Symbol,Nothing}=nothing)
    if idvar === nothing
        d = DataFrames.dropmissing(df[:, [var]], var)
        counts = DataFrames.combine(DataFrames.groupby(d, var), DataFrames.nrow => :Freq)
        DataFrames.sort!(counts, var)
        total = sum(counts.Freq)
        counts.Percent = 100 .* counts.Freq ./ total
        counts.Cum     = cumsum(counts.Percent)
        println(counts)
        return counts
    end

    d = DataFrames.dropmissing(df[:, [idvar, var]], [idvar, var])
    vals = sort(unique(d[!, var]))
    n_total  = DataFrames.nrow(d)
    N_panels = length(unique(d[!, idvar]))
    val_col = Any[]; freq_o = Int[]; pct_o = Float64[]
    freq_b  = Int[]; pct_b  = Float64[]; pct_w = Float64[]
    for v in vals
        mask_v = d[!, var] .== v
        push!(val_col, v)
        push!(freq_o, sum(mask_v))
        push!(pct_o,  100 * sum(mask_v) / n_total)
        panels_with_v = unique(d[mask_v, idvar])
        push!(freq_b, length(panels_with_v))
        push!(pct_b,  100 * length(panels_with_v) / N_panels)
        within = Float64[]
        for p in panels_with_v
            sub = d[d[!, idvar] .== p, :]
            push!(within, 100 * sum(sub[!, var] .== v) / DataFrames.nrow(sub))
        end
        push!(pct_w, isempty(within) ? 0.0 : Statistics.mean(within))
    end
    # Total row
    push!(val_col, "Total")
    push!(freq_o, n_total)
    push!(pct_o, 100.0)
    push!(freq_b, sum(freq_b))
    push!(pct_b,  sum(pct_b))
    push!(pct_w,  Statistics.mean(pct_w))

    out = DataFrames.DataFrame([
        var            => val_col,
        :Freq_over     => freq_o,
        :Percent_over  => pct_o,
        :Freq_btw      => freq_b,
        :Percent_btw   => pct_b,
        :Percent_within => pct_w,
    ])
    println("stata_xttab  —  $(var)   (N panels = $N_panels, total obs = $n_total)")
    println(out)
    return out
end

"""
    stata_xttrans(df, var, idvar, timevar; freq=false, varlabel=nothing) -> NamedTuple

Stata-style `stata_xttrans`. One-period transition matrix of `var` within
each panel. Default output is row-percentages only (matching Stata's
default); pass `freq=true` to also include the cell frequencies above
each percentage row.

`varlabel` may be a `String` shown as a multi-line column header
(Stata uses the variable label here when set); the helper auto-wraps
it across the leftmost column.
"""
function stata_xttrans(df, var::Symbol, idvar::Symbol, timevar::Symbol;
                 freq::Bool = false,
                 varlabel::Union{Nothing, AbstractString} = nothing)
    d = DataFrames.sort(df, [idvar, timevar])
    lagcol = Symbol(var, "_lag")
    d = DataFrames.transform(DataFrames.groupby(d, idvar),
        var => (x -> [missing; collect(x)[1:end-1]]) => lagcol)
    d = DataFrames.dropmissing(d, [lagcol, var])

    # Category levels in natural order (unwrap Stata labeled values).
    cats  = sort(unique(_sm_rawval.(d[!, var])))
    nc    = length(cats)

    # Transition count matrix  M[i, j] = # (lag = cats[i], current = cats[j]).
    M = zeros(Int, nc, nc)
    for i in 1:DataFrames.nrow(d)
        li = findfirst(==(_sm_rawval(d[i, lagcol])), cats)
        ci = findfirst(==(_sm_rawval(d[i, var])),    cats)
        (li === nothing || ci === nothing) && continue
        M[li, ci] += 1
    end

    row_sum = vec(sum(M;  dims = 2))
    col_sum = vec(sum(M;  dims = 1))
    N_tot   = sum(M)
    row_pct = 100.0 .* M        ./ max.(row_sum, 1)
    col_pct = 100.0 .* col_sum ./ max(N_tot, 1)

    _comma(x::Integer) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")
    _pct(x)            = Printf.@sprintf("%.2f", x)

    label_w = max(10, length(string(var)) + 2)       # leftmost column
    cell_w  = 10                                     # per-category column
    totw    = cell_w

    inner_w = nc * (cell_w + 1) + 1
    ruler   = "-"^label_w * "-+" * "-"^inner_w * "+" * "-"^(totw + 1)

    # ── Header. If `varlabel` is given, wrap it across `label_w`-1 chars
    # over up to 3 lines (Stata's stata_xttrans does this). The right side of
    # the header has the depvar label split similarly, but we render a
    # single-line "<var> = ..." next to the column header for simplicity.
    if varlabel !== nothing
        # Wrap the label into chunks no wider than label_w-1.
        words = split(varlabel)
        lines = String[]
        cur = ""
        for w in words
            cand = isempty(cur) ? w : cur * " " * w
            if length(cand) <= label_w - 1
                cur = cand
            else
                push!(lines, cur); cur = w
            end
        end
        isempty(cur) || push!(lines, cur)
        for (i, l) in enumerate(lines)
            line = lpad(l, label_w) * " |"
            if i == 1
                line *= "  " * varlabel
            end
            println(line)
        end
        # Header column row
        hdr = lpad("", label_w) * " |"
        for c in cats
            hdr *= lpad(string(c), cell_w + 1)
        end
        hdr *= " |" * lpad("Total", totw + 1)
        println(hdr)
    else
        # No `varlabel`: Stata's default stata_xttrans header puts the variable
        # NAME centered above the category columns on line 1, then uses
        # it as the row-label on line 2.
        line1 = lpad("", label_w) * " |" *
                lpad(string(var), div(inner_w + length(string(var)), 2))
        println(line1)
        hdr = lpad(string(var), label_w) * " |"
        for c in cats
            hdr *= lpad(string(c), cell_w + 1)
        end
        hdr *= " |" * lpad("Total", totw + 1)
        println(hdr)
    end
    println(ruler)

    # ── Data rows: one per lag category. Stata's default is row-pct only;
    # `freq=true` adds the count line above each pct line.
    for i in 1:nc
        if freq
            r = lpad(string(cats[i]), label_w) * " |"
            for j in 1:nc
                r *= lpad(_comma(M[i, j]), cell_w + 1)
            end
            r *= " |" * lpad(_comma(row_sum[i]), totw + 1)
            println(r)
            r = lpad("", label_w) * " |"
            for j in 1:nc
                r *= lpad(_pct(row_pct[i, j]), cell_w + 1)
            end
            r *= " |" * lpad("100.00", totw + 1)
            println(r)
            println(ruler)
        else
            r = lpad(string(cats[i]), label_w) * " |"
            for j in 1:nc
                r *= lpad(_pct(row_pct[i, j]), cell_w + 1)
            end
            r *= " |" * lpad("100.00", totw + 1)
            println(r)
        end
    end
    freq || println(ruler)

    # ── Total row (column %). With `freq=true`, also show column sums.
    if freq
        r = lpad("Total", label_w) * " |"
        for j in 1:nc
            r *= lpad(_comma(col_sum[j]), cell_w + 1)
        end
        r *= " |" * lpad(_comma(N_tot), totw + 1)
        println(r)
        r = lpad("", label_w) * " |"
        for j in 1:nc
            r *= lpad(_pct(col_pct[j]), cell_w + 1)
        end
        r *= " |" * lpad("100.00", totw + 1)
        println(r)
    else
        r = lpad("Total", label_w) * " |"
        for j in 1:nc
            r *= lpad(_pct(col_pct[j]), cell_w + 1)
        end
        r *= " |" * lpad("100.00", totw + 1)
        println(r)
    end

    return (; freq = M, row_pct = row_pct, col_sum = col_sum,
              row_sum = row_sum, col_pct = col_pct,
              categories = cats, N = N_tot)
end

"""
    stata_matlist(M, name="M"; symmetric=true)

Stata-style `matrix list` output for a numeric matrix. With `symmetric=true`
(default) only the lower triangle is printed — matching the look of
`matrix list e(R)` after `xtreg, pa`. Numbers use Stata's `%9.0g`-style
formatting (8 sig digits, leading `0.` stripped, integers shown without
decimals).
"""
function stata_matlist(M::AbstractMatrix, name::AbstractString = "M";
                 symmetric::Bool = true)
    n, m = size(M)
    _fmt(x) = begin
        (ismissing(x) || !isfinite(x)) && return "."
        if isinteger(x) && abs(x) < 1e12
            return string(Int(round(x)))
        end
        s = Printf.@sprintf("%.8g", x)
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        return s
    end
    prefix = symmetric ? "symmetric " : ""
    println("$(prefix)$(name)[$(n),$(m)]")
    cell_w = 11
    # Column header
    hdr = " "^3
    for j in 1:m
        hdr *= lpad("c$(j)", cell_w)
    end
    println(hdr)
    # Rows
    for i in 1:n
        row = rpad("r$(i)", 3)
        j_end = symmetric ? i : m
        for j in 1:j_end
            row *= lpad(_fmt(M[i, j]), cell_w)
        end
        println(row)
    end
    return nothing
end
