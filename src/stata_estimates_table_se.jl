# (deps provided by the StatEcon module)

"""
    stata_estimates_table_se(specs; stats=[:N, :ll], eq_label="#1",
        b_fmt="%.4f", se_fmt="%.4f", stat_fmt="%d",
        col_w=12, var_w=12,
        aux_rows=NamedTuple[], omitted_per_spec=Dict{String,Set{String}}(),
        stats_skip=Dict{String,Set{Symbol}}())

Stata-style `estimates table m1 … , se b(%8.4f) stats(N ll)`. Variant
of `estimates_table_stata` that prints **standard errors** below each
coefficient row (instead of t-stats) and supports the panel-logit
comparison layout:

- `eq_label` — single equation label printed once at the top of the
  main block (e.g. `"#1"` to mimic Stata's `equations(1)` option).
- `aux_rows` — optional vector of `(var_name, values::Dict)` pairs for
  separate equation rows (e.g. `RE`'s `/lnsig2u`). `values` maps spec
  name → `(estimate, std_err)`; specs not in the dict get a blank cell.
- `omitted_per_spec` — `Dict(spec_name => Set(var_names))` for cells
  that should print `(omitted)` (FE-style time-invariant regressors).
- `stats_skip` — `Dict(spec_name => Set(stat_symbols))` for stat cells
  to leave blank (e.g. `"PA" => Set([:ll])` because GEE has no
  likelihood).

Each spec is the same NamedTuple shape used by `estimates_table_stata`
(`name, β, se, coefnames, n, ll`). Returns `nothing`.
"""
function stata_estimates_table_se(specs::AbstractVector;
                                  stats::AbstractVector{Symbol} = [:N, :ll],
                                  eq_label::AbstractString = "#1",
                                  b_fmt::AbstractString = "%.4f",
                                  se_fmt::AbstractString = "%.4f",
                                  stat_fmt::AbstractString = "%d",
                                  col_w::Int = 12, var_w::Int = 12,
                                  aux_rows::AbstractVector = NamedTuple[],
                                  omitted_per_spec::Dict{String,Set{String}} =
                                      Dict{String,Set{String}}(),
                                  stats_skip::Dict{String,Set{Symbol}} =
                                      Dict{String,Set{Symbol}}())
    nice(v) = v == "(Intercept)" ? "_cons" : v
    nc = length(specs)
    line_w = var_w + 2 + nc * col_w

    # Build the union of main equation variables (in order of first appearance).
    main_vars = String[]
    for s in specs, v in s.coefnames
        nv = nice(v)
        nv in main_vars || push!(main_vars, nv)
    end

    # Cell printers: lpad value into (col_w − 2) chars + 2 trailing spaces.
    pad = col_w - 2
    blank() = " " ^ col_w
    cell(s::AbstractString) = lpad(s, pad) * "  "
    cellv(x, fmt) = cell(Printf.format(Printf.Format(fmt), x))

    println("-" ^ line_w)
    # Header: "Variable | <name1> <name2> ..."
    print(lpad("Variable", var_w), " |")
    for c in specs
        print(cell(c.name))
    end
    println()
    println("-" ^ var_w, "+", "-" ^ (line_w - var_w - 1))

    # Main equation block label, e.g. "#1".
    println(rpad(eq_label, var_w), " |")
    for v in main_vars
        # Coefficient row.
        print(lpad(v, var_w), " |")
        for c in specs
            cnames = nice.(c.coefnames)
            i = findfirst(==(v), cnames)
            is_omitted = haskey(omitted_per_spec, c.name) &&
                         v in omitted_per_spec[c.name]
            if is_omitted
                print(cell("(omitted)"))
            elseif i === nothing
                print(blank())
            else
                print(cellv(c.β[i], b_fmt))
            end
        end
        println()
        # Std. err. row.
        print(" " ^ var_w, " |")
        for c in specs
            cnames = nice.(c.coefnames)
            i = findfirst(==(v), cnames)
            is_omitted = haskey(omitted_per_spec, c.name) &&
                         v in omitted_per_spec[c.name]
            if is_omitted || i === nothing
                print(blank())
            else
                print(cellv(c.se[i], se_fmt))
            end
        end
        println()
    end

    # Auxiliary equation rows (e.g. /lnsig2u — present only for RE).
    for aux in aux_rows
        println("-" ^ var_w, "+", "-" ^ (line_w - var_w - 1))
        # Coef
        print(lpad(aux.var_name, var_w), " |")
        for c in specs
            if haskey(aux.values, c.name)
                val, _ = aux.values[c.name]
                print(cellv(val, b_fmt))
            else
                print(blank())
            end
        end
        println()
        # SE
        print(" " ^ var_w, " |")
        for c in specs
            if haskey(aux.values, c.name)
                _, se_v = aux.values[c.name]
                print(cellv(se_v, se_fmt))
            else
                print(blank())
            end
        end
        println()
    end

    # Statistics block.
    println("-" ^ var_w, "+", "-" ^ (line_w - var_w - 1))
    println(rpad("Statistics", var_w), " |")
    for st in stats
        print(lpad(string(st), var_w), " |")
        for c in specs
            skip = haskey(stats_skip, c.name) && st in stats_skip[c.name]
            if skip
                print(blank())
            elseif st == :N
                print(cellv(c.n, "%d"))
            elseif st == :ll
                print(cellv(c.ll, stat_fmt))
            else
                val = getfield(c, st)
                print(cellv(val, stat_fmt))
            end
        end
        println()
    end
    println("-" ^ line_w)
    return nothing
end

