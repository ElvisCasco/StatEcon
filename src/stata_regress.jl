# (deps provided by the StatEcon module)

# Reproduces Stata's `regress y x, vce(robust)` output for a fitted
# FixedEffectModels model (also works for a GLM lm via its coeftable).
#
#   ols = FixedEffectModels.reg(df, FixedEffectModels.@formula(D_lwage ~ train),
#                               FixedEffectModels.Vcov.robust())
#
#   stata_regress(ols)                                   # names taken from the model
#   stata_regress(ols; yname = "D.lwage")                 # rename the dependent variable
#   stata_regress(ols; xnames = ["train"])                # rename regressors, in order
#   stata_regress(ols; xnames = Dict("train" => "D.train"))          # rename by name
#   stata_regress(ols; yname = "D.lwage", xnames = ["train"], vce = :robust)
#
# Arguments
#   yname  : display name of the dependent variable (default: the model's response).
#   xnames : display names of the regressors. Either a vector applied positionally to
#            the non-intercept coefficients, or a Dict/NamedTuple/Pairs mapping the
#            model's coefficient name to the label to show. `_cons` is never renamed.
#   vce    : `vce(...)` option -- :robust, :cluster or :ols. Controls the "Robust"
#            banner over the std. err. column. Default `nothing` infers it from the
#            model's own vcov (so `Vcov.robust()`/`Vcov.cluster(...)` are detected).
#   title  : header title (default "Linear regression").
#
# Prints the header block (Number of obs / F / Prob>F / R-squared / Root MSE)
# and the coefficient table (Coefficient, robust std. err., t, P>|t|, 95% CI),
# renaming the intercept to `_cons` and placing it last, as Stata does.

# Build a name -> label map from a Dict / NamedTuple / vector of Pairs.
# Returns `nothing` when `xnames` is a plain vector (applied positionally instead).
# Julia columns built from Stata time-series operators are named D_x / L_x / F_x;
# display them the Stata way (D.x / L.x / F.x). Disable with `tsops = false`.
_sr_tsop(s) = replace(String(s), r"^([DLF])_" => s"\1.")

# Coefficient label: FixedEffectModels writes interactions as "a & b", Stata as
# "c.a#c.b" (continuous x continuous). Disable with `interact = false`.
function _sr_label(s, tsops::Bool, interact::Bool)
    parts = split(String(s), r"\s*&\s*")
    f(p) = tsops ? _sr_tsop(strip(p)) : String(strip(p))
    length(parts) == 1 && return f(parts[1])
    interact || return join(f.(parts), " & ")
    return join(["c." * f(p) for p in parts], "#")
end

function _sr_labelmap(xnames)
    xnames isa AbstractDict && return Dict(String(k) => String(v) for (k, v) in xnames)
    xnames isa NamedTuple   && return Dict(String(k) => String(v) for (k, v) in pairs(xnames))
    if xnames isa AbstractVector && !isempty(xnames) && first(xnames) isa Pair
        return Dict(String(p.first) => String(p.second) for p in xnames)
    end
    return nothing
end

# Stata number for the header (%g-style, `sig` significant figures, leading 0 dropped).
function _sr_num(x::Real, sig::Int = 7)
    isnan(x) && return "."
    s = @sprintf("%.*g", sig, x)
    s = replace(s, r"^(-?)0\." => s"\g<1>.")
    return s
end

# Stata coefficient-table format: 7 decimal places, trailing zeros stripped,
# leading zero suppressed (matches Stata's %9.0g here: .0474800 -> .04748).
function _sr_coef(x::Real)
    isnan(x) && return "."
    s = @sprintf("%.7f", x)
    occursin('.', s) && (s = rstrip(s, '0'); s = rstrip(s, '.'))
    s = replace(s, r"^(-?)0\." => s"\g<1>.")
    return s
end

# Build a Vcov estimator from the `vce(...)` option.
#   :ols / :simple -> Vcov.simple()   :robust -> Vcov.robust()
#   :cluster       -> Vcov.cluster(cluster...)   (needs `cluster = :id`)
# A Vcov object may also be passed straight through.
function _sr_vcov(vce, cluster)
    (vce isa Symbol || vce isa AbstractString) || return vce
    s = lowercase(String(vce))
    occursin("robust", s) && return FixedEffectModels.Vcov.robust()
    if occursin("cluster", s)
        isnothing(cluster) && error("stata_regress: vce = :cluster requires `cluster = :id`")
        return cluster isa Symbol ? FixedEffectModels.Vcov.cluster(cluster) :
                                    FixedEffectModels.Vcov.cluster(cluster...)
    end
    return FixedEffectModels.Vcov.simple()
end

# Fit and report in one call -- the regression is run inside stata_regress, so no
# separate `ols = FixedEffectModels.reg(...)` step is needed. Returns the fitted
# model, so `residuals`/`predict`/`coef` on the result still work.
#
#   ols = stata_regress(df, FixedEffectModels.@formula(D_lwage ~ train);
#                       yname = "D.lwage", xnames = ["train" => "train"],
#                       vce = :robust, title = "Linear regression")
#
#   ols = stata_regress(df, term(:lwage) ~ term(:train) + fe(:id);
#                       vce = :cluster, cluster = :id)
function stata_regress(df, formula; vce = :ols, cluster = nothing, kwargs...)
    m = FixedEffectModels.reg(df, formula, _sr_vcov(vce, cluster))
    stata_regress(m; vce = vce, kwargs...)
    return m
end

# One regressor -> a term. `:train` is a plain variable; a tuple `(:train, :educ_dm)`
# is Stata's interaction c.train#c.educ_dm.
_sr_term(v::Symbol)          = StatsModels.Term(v)
_sr_term(v::Tuple)           = reduce(&, StatsModels.Term.(v))
_sr_term(v::AbstractVector)  = reduce(&, StatsModels.Term.(v))
_sr_term(v)                  = v          # already a term

# Stata-style call: name the dependent variable and the regressors directly,
# just like `reg y x1 x2, vce(robust)` -- no formula needed.
#
#   ols = stata_regress(df, y = :D_lwage, x = [:train], vce = :robust)
#   ols = stata_regress(df, y = :lwage,  x = [:train, :f92, :d],
#                       vce = :cluster, cluster = :id)
#   ols = stata_regress(df, y = :D_lwage,                       # interactions
#                       x = [:train, (:train, :educ_dm), (:train, :exper_dm),
#                            :d92, :educ, :exper], nocons = true, vce = :robust)
#   ols = stata_regress(df, y = :lwage, x = [:train, :f92],     # xtreg ..., fe
#                       absorb = :id, vce = :cluster, cluster = :id)
#
#   y       : dependent variable (Symbol).
#   x       : regressors -- Symbols, or tuples for interactions.
#   nocons  : Stata's `nocons` (drop the constant).
#   absorb  : fixed effect(s) to absorb, i.e. Stata's `xtreg ..., fe`.
function stata_regress(df::AbstractDataFrame; y::Symbol, x, vce = :ols,
                       cluster = nothing, absorb = nothing, nocons::Bool = false,
                       kwargs...)
    rhs = Any[nocons ? StatsModels.ConstantTerm(0) : StatsModels.ConstantTerm(1)]
    for v in x
        push!(rhs, _sr_term(v))
    end
    if absorb !== nothing
        for a in (absorb isa Symbol ? (absorb,) : absorb)
            push!(rhs, FixedEffectModels.fe(a))
        end
    end
    formula = StatsModels.Term(y) ~ reduce(+, rhs)
    m = FixedEffectModels.reg(df, formula, _sr_vcov(vce, cluster))
    stata_regress(m; vce = vce, kwargs...)
    return m
end

function stata_regress(m; yname::Union{Nothing,AbstractString} = nothing,
                       xnames = nothing, vce = nothing, tsops::Bool = true,
                       interact::Bool = true,
                       title::Union{Nothing,AbstractString} = nothing)
    ct   = FixedEffectModels.coeftable(m)
    nm   = String.(ct.rownms)
    est  = ct.cols[1]; se = ct.cols[2]; tv = ct.cols[3]
    pv   = ct.cols[4]; lo = ct.cols[5]; hi = ct.cols[6]

    # order: non-intercept coefficients first, intercept ("(Intercept)") last as _cons
    isint(s) = occursin("Intercept", s)
    ord = vcat(findall(!isint, nm), findall(isint, nm))
    raw = [isint(nm[i]) ? "_cons" : nm[i] for i in ord]     # model coefficient names
    # Stata-style names: D_lwage -> D.lwage, "train & educ_dm" -> c.train#c.educ_dm
    labels = [l == "_cons" ? l : _sr_label(l, tsops, interact) for l in raw]

    # ---- optional renaming of the regressors (xnames) -------------------
    if xnames !== nothing
        lblmap = _sr_labelmap(xnames)
        if lblmap === nothing                    # plain vector: positional
            xs = String.(collect(xnames))
            k = 0
            for i in eachindex(labels)
                labels[i] == "_cons" && continue
                k += 1
                k <= length(xs) && (labels[i] = xs[k])
            end
        else                                     # Dict/NamedTuple/Pairs: by name.
            # Key on the raw model name ("train & educ_dm") or, failing that, on
            # the Stata-style label ("c.train#c.educ_dm"); otherwise keep as is.
            labels = [raw[i] == "_cons" ? "_cons" :
                      get(lblmap, raw[i], get(lblmap, labels[i], labels[i]))
                      for i in eachindex(raw)]
        end
    end

    N    = round(Int, FixedEffectModels.nobs(m))
    df2  = round(Int, FixedEffectModels.dof_residual(m))
    df1  = count(!isint, nm)
    R2   = FixedEffectModels.r2(m)
    rmse = sqrt(m.rss / FixedEffectModels.dof_residual(m))
    # absorbed fixed effects -> Stata's `xtreg ..., fe` (within) regression
    hasfe = hasproperty(m, :dof_fes) && m.dof_fes > 0
    ttl   = isnothing(title) ? (hasfe ? "Fixed-effects (within) regression" :
                                        "Linear regression") : String(title)
    dep  = isnothing(yname) ?
           (tsops ? _sr_tsop(m.responsename) : String(m.responsename)) : String(yname)
    # vce(...): default infers from the model's own vcov; :robust/:cluster/:ols override
    vt   = isnothing(vce) ? lowercase(string(m.vcov_type)) : lowercase(string(vce))
    robust    = occursin("robust", vt) || occursin("cluster", vt)
    clustered = occursin("cluster", vt)

    # ---- header block --------------------------------------------------
    hdr = [("Number of obs", @sprintf("%d", N)),
           (@sprintf("F(%d, %d)", df1, df2), @sprintf("%.2f", m.F)),
           ("Prob > F",  @sprintf("%.4f", m.p)),
           ("R-squared", @sprintf("%.4f", R2)),
           ("Root MSE",  _sr_num(rmse, 5))]
    # Stata's xtreg,fe also reports the within R-squared
    if hasfe && hasproperty(m, :r2_within) && m.r2_within !== nothing
        insert!(hdr, 5, ("R-sq: within", @sprintf("%.4f", m.r2_within)))
    end
    lw = maximum(length(h[1]) for h in hdr)
    vw = maximum(length(h[2]) for h in hdr)
    println(rpad(ttl, 48) * rpad(hdr[1][1], lw) * " = " * lpad(hdr[1][2], vw))
    for k in 2:length(hdr)
        println(" "^48 * rpad(hdr[k][1], lw) * " = " * lpad(hdr[k][2], vw))
    end
    # Stata's clustered-SE note, e.g. (Std. err. adjusted for 545 clusters in id)
    if clustered && hasproperty(m, :nclusters) && m.nclusters !== nothing
        for (cv, nc) in pairs(m.nclusters)
            println()
            println(@sprintf("(Std. err. adjusted for %d clusters in %s)", nc, cv))
        end
    end

    # ---- coefficient table ---------------------------------------------
    ests = [_sr_coef(est[i]) for i in ord]; ses = [_sr_coef(se[i]) for i in ord]
    tvs  = [@sprintf("%.2f", tv[i]) for i in ord]
    pvs  = [@sprintf("%.3f", pv[i]) for i in ord]
    los  = [_sr_coef(lo[i]) for i in ord]; his = [_sr_coef(hi[i]) for i in ord]

    namew = max(maximum(length, labels), length(dep), 8)
    c1 = max(maximum(length, ests), length("Coefficient"))
    c2 = max(maximum(length, ses),  length("std. err."))
    c3 = max(maximum(length, tvs),  4)
    c4 = max(maximum(length, pvs),  5)
    c5 = max(maximum(length, los),  length("[95% conf."))
    c6 = max(maximum(length, his),  length("interval]"))
    right = c1 + c2 + c3 + c4 + c5 + c6 + 12          # widths + spacing
    rule() = println("-"^(namew + 2 + right))         # full-width plain rule
    dash() = println("-"^(namew + 1) * "+" * "-"^right)  # rule with column junction

    rule()                                             # header/table divider (top)
    if robust                                          # "Robust" over std. err.
        println(" "^(namew + 1) * "| " * " "^c1 * "   " * lpad("Robust", c2))
    end
    println(lpad(dep, namew) * " | " * lpad("Coefficient", c1) * "  " *
            lpad("std. err.", c2) * "  " * lpad("t", c3) * "  " *
            lpad("P>|t|", c4) * "   " * lpad("[95% conf.", c5) * "  " *
            lpad("interval]", c6))
    dash()                                             # under column headers (with +)
    for i in eachindex(labels)
        println(lpad(labels[i], namew) * " | " * lpad(ests[i], c1) * "  " *
                lpad(ses[i], c2) * "  " * lpad(tvs[i], c3) * "  " *
                lpad(pvs[i], c4) * "   " * lpad(los[i], c5) * "  " * lpad(his[i], c6))
    end
    rule()                                             # bottom rule (plain)
    return m
end
