# (deps provided by the StatEcon module)

# Reproduces Stata's `ereturn list` for a fitted regression object.
#
#   ols = stata_regress(df, y = :mpg, x = [:price, :weight])
#   ereturn_list(ols)
#
# Prints the standard e()-scalars (N, df_m, df_r, r2, r2_a, rss, rmse, F, p),
# e()-macros (depvar, indeps, cmd) and the e(b) / e(V) matrices with the same
# labelling Stata uses. Returns everything in a NamedTuple so it can also be
# consumed programmatically.
#
# Works with any StatsBase-compatible model (FixedEffectModels, GLM, ...).
# Fields that aren't defined for a given model type are silently skipped, as
# `ereturn list` does in Stata.

_er_try(f, m) = try f(m) catch; missing end

# Pretty-print a numeric matrix with column (and optional row) labels, using
# only Printf -- avoids the PrettyTables dependency.
function _er_print_matrix(M::AbstractMatrix; colnames = nothing, rownames = nothing)
    m, n = size(M)
    colnames = colnames === nothing ? ["c$(j)" for j in 1:n] : String.(colnames)
    rownames = rownames === nothing ? String[] : String.(rownames)
    cellstrs = [Printf.@sprintf("%.6g", M[i, j]) for i in 1:m, j in 1:n]
    colw = [max(length(colnames[j]),
                maximum(length(cellstrs[i, j]) for i in 1:m; init = 0)) for j in 1:n]
    rowlblw = isempty(rownames) ? 0 : maximum(length.(rownames))

    header = (isempty(rownames) ? "" : " "^(rowlblw + 2)) *
             join((lpad(colnames[j], colw[j]) for j in 1:n), "  ")
    println(header)
    println("-"^length(header))
    for i in 1:m
        prefix = isempty(rownames) ? "" : rpad(rownames[i], rowlblw) * "  "
        println(prefix *
                join((lpad(cellstrs[i, j], colw[j]) for j in 1:n), "  "))
    end
end

function ereturn_list(m)
    # ---- e()-scalars -------------------------------------------------------
    rss  = _er_try(deviance, m)
    dfr  = _er_try(dof_residual, m)
    rmse = (ismissing(rss) || ismissing(dfr) || dfr == 0) ? missing : sqrt(rss / dfr)

    scalars = (;
        N     = _er_try(nobs, m),
        df_m  = _er_try(dof, m),
        df_r  = dfr,
        r2    = _er_try(r2, m),
        r2_a  = _er_try(adjr2, m),
        rss   = rss,
        rmse  = rmse,
        F     = hasproperty(m, :F) ? m.F : missing,
        p     = hasproperty(m, :p) ? m.p : missing,
    )

    # ---- e()-macros --------------------------------------------------------
    depvar = hasproperty(m, :yname)        ? string(m.yname) :
             hasproperty(m, :responsename) ? string(m.responsename) : missing
    indeps = try join(string.(StatsBase.coefnames(m)), " ") catch; missing end
    macros = (; depvar = depvar, indeps = indeps,
                cmd    = string(nameof(typeof(m))))

    # ---- e(b) / e(V) -------------------------------------------------------
    b = _er_try(coef, m)
    V = _er_try(vcov, m)

    # ---- print in Stata's ereturn layout ----------------------------------
    println("scalars:")
    for (k, v) in pairs(scalars)
        ismissing(v) && continue
        println("  e($k) = ", v isa Real ? round(v; sigdigits = 8) : v)
    end
    println("\nmacros:")
    for (k, v) in pairs(macros)
        ismissing(v) && continue
        println("  e($k) : \"", v, "\"")
    end
    println("\nmatrices:")
    if !ismissing(b)
        names_ = try StatsBase.coefnames(m) catch; ["x$i" for i in eachindex(b)] end
        println("  e(b)  : 1 x $(length(b))")
        _er_print_matrix(reshape(b, 1, :); colnames = names_)
    end
    if !ismissing(V)
        names_ = try StatsBase.coefnames(m) catch; ["x$i" for i in 1:size(V, 1)] end
        println("  e(V)  : $(size(V, 1)) x $(size(V, 2))")
        _er_print_matrix(V; colnames = names_, rownames = names_)
    end

    return (; scalars, macros, b, V)
end
