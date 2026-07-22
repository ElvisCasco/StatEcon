# --------------------------------------------------------------------------
# Linear instrumental-variables estimators (Cameron & Trivedi ch. 6):
#   stata_ivregress_2sls  — 2SLS (first-stage + 2SLS display)
#   stata_ivregress_gmm   — IV/GMM (:twosls or :gmm; :classical/:robust/:cluster W)
#   stata_liml            — LIML
#   stata_jive            — Jackknife IV (JIVE1)
#   stata_estimates_table — Stata-style side-by-side comparison table
# --------------------------------------------------------------------------

# Comma-formatter for "Number of obs" display
_ivr_fmtn(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")

# Format a real number to a Stata-style compact string (used in coef cells).
_ivr_g(x::Real, w::Int; d::Int = 7) =
    isnan(x) || !isfinite(x) ? Printf.@sprintf("%*s", w, ".") :
    Printf.@sprintf("%*.*f", w, d, x)

# Print the coefficient block. `mode` = :t (OLS) or :z (2SLS/IV).
function _ivr_print_coef_block(dep::AbstractString, nm::Vector{String},
                               β::Vector{Float64}, se::Vector{Float64},
                               order::Vector{Int}; mode::Symbol,
                               df_r::Int, level::Float64,
                               robust::Bool)
    lvl = round(Int, 100 * level)
    zt_label = mode === :t ? "t" : "z"
    p_label  = mode === :t ? "P>|t|" : "P>|z|"
    se_hdr   = robust ? "Robust std. err." : "Std. err."

    stat = β ./ se
    if mode === :t
        crit  = Distributions.quantile(Distributions.TDist(df_r), 1 - (1 - level) / 2)
        pvals = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(df_r), abs.(stat)))
    else
        crit  = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
        pvals = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(stat)))
    end
    lo = β .- crit .* se
    hi = β .+ crit .* se

    headers = [dep, "Coefficient", se_hdr, zt_label, p_label,
               "[$(lvl)% conf.", "interval]"]
    rows = Vector{Vector{String}}()
    for i in order
        push!(rows, [nm[i],
                     Printf.@sprintf("%.7f", β[i]),
                     Printf.@sprintf("%.7f", se[i]),
                     Printf.@sprintf("%.2f", stat[i]),
                     Printf.@sprintf("%.3f", pvals[i]),
                     Printf.@sprintf("%.7f", lo[i]),
                     Printf.@sprintf("%.7f", hi[i])])
    end
    _sug_print_table(headers, rows,
                     [:l, :r, :r, :r, :r, :r, :r])
end

"""
    stata_ivregress_2sls(df, y, endog_iv::Pair, exog=Symbol[];
                        robust=true, level=0.95, first=true)

Stata-style `ivregress 2sls y exog (endog = iv), vce(robust)` with optional
first-stage regressions. `endog_iv = :endog => :iv` (or vectors of Symbols).
Returns `(; first_stage, iv)` where `iv` is the `FixedEffectModels.reg`
model object.
"""
function stata_ivregress_2sls(df, y, endog_iv::Pair, exog = Symbol[];
                              robust::Bool = true, level::Float64 = 0.95,
                              first::Bool = true)
    ys    = Symbol(y)
    endog = endog_iv.first isa Union{Symbol,AbstractString} ?
            [Symbol(endog_iv.first)] :
            [Symbol(v) for v in endog_iv.first]
    iv    = endog_iv.second isa Union{Symbol,AbstractString} ?
            [Symbol(endog_iv.second)] :
            [Symbol(v) for v in endog_iv.second]
    exog_v = isempty(exog) ? Symbol[] :
             (exog isa Union{Symbol,AbstractString} ?
                 [Symbol(exog)] : [Symbol(v) for v in exog])
    vcov = robust ? FixedEffectModels.Vcov.robust() :
                    FixedEffectModels.Vcov.simple()
    lvl = round(Int, 100 * level)

    _print_first(m, depname) = begin
        β    = StatsBase.coef(m); V = StatsBase.vcov(m)
        se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
        nm   = replace.(string.(StatsBase.coefnames(m)), "(Intercept)" => "_cons")
        n    = Int(StatsBase.nobs(m))
        k    = length(β)
        df_r = Int(StatsBase.dof_residual(m))
        rss  = StatsBase.deviance(m)
        r2v  = StatsBase.r2(m)
        rmse = sqrt(rss / df_r)
        cons_idx = findfirst(==("_cons"), nm)
        slope    = cons_idx === nothing ? collect(1:k) : setdiff(1:k, [cons_idx])
        q        = length(slope)
        Wstat    = β[slope]' * LinearAlgebra.inv(V[slope, slope]) * β[slope]
        Fstat    = Wstat / q
        pF       = 1 - Distributions.cdf(Distributions.FDist(q, df_r), Fstat)
        ar2      = 1 - (1 - r2v) * (n - 1) / df_r
        Printf.@printf("%-56s%-14s= %6s\n",  "", "Number of obs", _ivr_fmtn(n))
        Printf.@printf("%-56s%-14s= %6.2f\n","", "F($q, $df_r)", Fstat)
        Printf.@printf("%-56s%-14s= %6.4f\n","", "Prob > F",      pF)
        Printf.@printf("%-56s%-14s= %6.4f\n","", "R-squared",     r2v)
        Printf.@printf("%-56s%-14s= %6.4f\n","", "Adj R-squared", ar2)
        Printf.@printf("%-56s%-14s= %6.4f\n","", "Root MSE",      rmse)
        println()
        order = cons_idx === nothing ? slope : vcat(slope, cons_idx)
        _ivr_print_coef_block(depname, nm, β, se, order;
                              mode = :t, df_r = df_r,
                              level = level, robust = robust)
    end

    _print_second(m, depname, endog_nms) = begin
        β    = StatsBase.coef(m); V = StatsBase.vcov(m)
        se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
        nm   = replace.(string.(StatsBase.coefnames(m)), "(Intercept)" => "_cons")
        n    = Int(StatsBase.nobs(m))
        k    = length(β)
        df_r = Int(StatsBase.dof_residual(m))
        rss  = StatsBase.deviance(m)
        r2v  = StatsBase.r2(m)
        rmse = sqrt(rss / df_r)
        cons_idx = findfirst(==("_cons"), nm)
        slope    = cons_idx === nothing ? collect(1:k) : setdiff(1:k, [cons_idx])
        q        = length(slope)
        Wstat    = β[slope]' * LinearAlgebra.inv(V[slope, slope]) * β[slope]
        pW       = 1 - Distributions.cdf(Distributions.Chisq(q), Wstat)
        Printf.@printf("%-50s%-16s= %10s\n",
                       "Instrumental-variables 2SLS regression",
                       "Number of obs", _ivr_fmtn(n))
        Printf.@printf("%-50s%-16s= %10.2f\n", "", "Wald chi2($q)", Wstat)
        Printf.@printf("%-50s%-16s= %10.4f\n", "", "Prob > chi2",   pW)
        Printf.@printf("%-50s%-16s= %10.4f\n", "", "R-squared",     r2v)
        Printf.@printf("%-50s%-16s= %10.4f\n", "", "Root MSE",      rmse)
        println()
        endog_pos = Int[]
        for ev in endog_nms
            i = findfirst(==(string(ev)), nm)
            i !== nothing && push!(endog_pos, i)
        end
        rest = setdiff(slope, endog_pos)
        order = cons_idx === nothing ?
                vcat(endog_pos, rest) :
                vcat(endog_pos, rest, cons_idx)
        _ivr_print_coef_block(depname, nm, β, se, order;
                              mode = :z, df_r = df_r,
                              level = level, robust = robust)
    end

    # First-stage regressions
    first_stage_models = Any[]
    if first
        println("First-stage regressions")
        println("-"^23); println()
        rhs = vcat(exog_v, iv)
        for ev in endog
            f_fs = term(ev) ~ sum(term.(rhs))
            m_fs = FixedEffectModels.reg(df, f_fs, vcov)
            _print_first(m_fs, string(ev))
            println()
            push!(first_stage_models, m_fs)
        end
    end

    # 2SLS via @formula
    exog_part  = isempty(exog_v) ? "1" : join(string.(exog_v), " + ")
    endog_part = join(string.(endog), " + ")
    iv_part    = join(string.(iv), " + ")
    fmla_str   = "@formula($(string(ys)) ~ $exog_part + ($endog_part ~ $iv_part))"
    f_iv       = Base.eval(@__MODULE__, Meta.parse(fmla_str))
    m_iv       = FixedEffectModels.reg(df, f_iv, vcov)
    _print_second(m_iv, string(ys), endog)
    Printf.@printf("Endogenous: %s\n", join(string.(endog), " "))
    Printf.@printf("Exogenous:  %s\n",
                   join(string.(vcat(exog_v, iv)), " "))
    return (; first_stage = first_stage_models, iv = m_iv)
end

"""
    stata_ivregress_gmm(df, y, endog, instruments, exog=Symbol[];
                       estimator=:twosls, wmatrix=:classical,
                       cluster=nothing, iterate=false,
                       max_iter=100, tol=1e-9)

Linear IV by 2SLS or GMM. Returns `(; β, V, se, coefnames, n)`.
`coefnames` is `["_cons", endog…, exog…]`.

`estimator`:
  * `:twosls` — β̂ with W = (Z'Z)⁻¹;
  * `:gmm`   — β̂ with W = Ω̂⁻¹.

`wmatrix` selects both the point weight (for `:gmm`) and the reported
vcov: `:classical`, `:robust`, or `:cluster` (with `cluster` Symbol).
`iterate=true` runs iterated GMM until `‖β − β_prev‖ < tol`.
"""
function stata_ivregress_gmm(df, y, endog, instruments, exog = Symbol[];
                             estimator::Symbol = :twosls,
                             wmatrix::Symbol = :classical,
                             cluster::Union{Symbol,Nothing} = nothing,
                             iterate::Bool = false,
                             max_iter::Int = 100, tol::Float64 = 1e-9)
    ys      = Symbol(y)
    endog_v = endog isa Union{Symbol,AbstractString} ?
              [Symbol(endog)] : [Symbol(v) for v in endog]
    iv_v    = instruments isa Union{Symbol,AbstractString} ?
              [Symbol(instruments)] : [Symbol(v) for v in instruments]
    exog_v  = isempty(exog) ? Symbol[] :
              (exog isa Union{Symbol,AbstractString} ?
                  [Symbol(exog)] : [Symbol(v) for v in exog])

    needed = vcat(ys, endog_v, exog_v, iv_v)
    cluster !== nothing && push!(needed, cluster)
    d = DataFrames.dropmissing(df, unique(needed))
    n = DataFrames.nrow(d)

    y_vec     = Float64.(_sm_rawval.(d[!, ys]))
    endog_mat = hcat([Float64.(_sm_rawval.(d[!, v])) for v in endog_v]...)
    exog_mat  = isempty(exog_v) ? zeros(n, 0) :
                hcat([Float64.(_sm_rawval.(d[!, v])) for v in exog_v]...)
    iv_mat    = hcat([Float64.(_sm_rawval.(d[!, v])) for v in iv_v]...)
    cl_vec    = cluster === nothing ? nothing : _sm_rawval.(d[!, cluster])

    X = hcat(ones(n), endog_mat, exog_mat)
    Z = hcat(ones(n), exog_mat,  iv_mat)

    _point(y_, X_, Z_, W) = (XZ = X_'Z_; (XZ * W * XZ') \ (XZ * W * (Z_'y_)))
    _sand = function (Z_, u, cl)
        m = size(Z_, 2); Ω = zeros(m, m)
        if cl === nothing
            for i in 1:length(u)
                zi = @view Z_[i, :]
                Ω .+= (u[i]^2) .* (zi * zi')
            end
        else
            for g in unique(cl)
                sel = cl .== g
                zu  = Z_[sel, :]' * u[sel]
                Ω  .+= zu * zu'
            end
        end
        return Ω
    end
    _vcov(X_, Z_, W, Ω) = begin
        A_inv = LinearAlgebra.inv(X_'Z_ * W * Z_'X_)
        A_inv * (X_'Z_ * W * Ω * W * Z_'X_) * A_inv
    end

    if estimator == :twosls
        W = LinearAlgebra.inv(Z' * Z)
        β = _point(y_vec, X, Z, W)
        u = y_vec .- X * β
        if wmatrix == :classical
            σ² = sum(abs2, u) / (n - size(X, 2))
            V  = σ² .* LinearAlgebra.inv(X'Z * W * Z'X)
        else
            cl_for = wmatrix == :cluster ? cl_vec : nothing
            V      = _vcov(X, Z, W, _sand(Z, u, cl_for))
        end
    elseif estimator == :gmm
        cl_for = wmatrix == :cluster ? cl_vec : nothing
        W0 = LinearAlgebra.inv(Z' * Z)
        β  = _point(y_vec, X, Z, W0)
        u  = y_vec .- X * β
        Ω  = _sand(Z, u, cl_for)
        W  = LinearAlgebra.inv(Ω)
        β  = _point(y_vec, X, Z, W)
        u  = y_vec .- X * β
        if iterate
            for _ in 1:max_iter
                Ω_new = _sand(Z, u, cl_for)
                β_new = _point(y_vec, X, Z, LinearAlgebra.inv(Ω_new))
                u_new = y_vec .- X * β_new
                LinearAlgebra.norm(β_new - β) < tol &&
                    (β = β_new; u = u_new; break)
                β, u = β_new, u_new
            end
        end
        Ω = _sand(Z, u, cl_for)
        W = LinearAlgebra.inv(Ω)
        V = _vcov(X, Z, W, Ω)
    else
        error("Unknown estimator: $estimator (expected :twosls or :gmm)")
    end

    se        = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    coefnames = vcat("_cons", string.(endog_v), string.(exog_v))
    return (; β, V, se, coefnames, n)
end

"""
    stata_liml(df, y, endog, instruments, exog=Symbol[]; robust=true)

Limited Information Maximum Likelihood (LIML) IV. Returns
`(; β, V, se, coefnames, n, κ)`.
"""
function stata_liml(df, y, endog, instruments, exog = Symbol[];
                    robust::Bool = true)
    ys      = Symbol(y)
    endog_v = endog isa Union{Symbol,AbstractString} ?
              [Symbol(endog)] : [Symbol(v) for v in endog]
    inst_v  = instruments isa Union{Symbol,AbstractString} ?
              [Symbol(instruments)] : [Symbol(v) for v in instruments]
    exog_v  = isempty(exog) ? Symbol[] :
              (exog isa Union{Symbol,AbstractString} ?
                   [Symbol(exog)] : [Symbol(v) for v in exog])

    needed = unique(vcat(ys, endog_v, exog_v, inst_v))
    d = DataFrames.dropmissing(df, needed)
    n = DataFrames.nrow(d)
    Y = Float64.(_sm_rawval.(d[!, ys]))
    X_endog = hcat([Float64.(_sm_rawval.(d[!, v])) for v in endog_v]...)
    X_exog  = isempty(exog_v) ? zeros(n, 0) :
              hcat([Float64.(_sm_rawval.(d[!, v])) for v in exog_v]...)
    Z_excl  = hcat([Float64.(_sm_rawval.(d[!, v])) for v in inst_v]...)

    X_all    = hcat(ones(n), X_endog, X_exog)
    X_exog_c = hcat(ones(n), X_exog)
    Z_full   = hcat(ones(n), X_exog, Z_excl)

    W_    = hcat(Y, X_endog)
    WtW   = W_' * W_
    XeInv = LinearAlgebra.inv(X_exog_c' * X_exog_c)
    ZtZi  = LinearAlgebra.inv(Z_full' * Z_full)
    WtXe  = W_' * X_exog_c
    WtZ   = W_' * Z_full
    A     = WtW - WtXe * XeInv * WtXe'
    B     = WtW - WtZ  * ZtZi  * WtZ'
    κ     = minimum(real.(LinearAlgebra.eigvals(A, B)))

    ZtX  = Z_full' * X_all
    PZ_X = Z_full * (ZtZi * ZtX)
    KX   = (1 - κ) .* X_all .+ κ .* PZ_X
    ZtY  = Z_full' * Y
    PZ_Y = Z_full * (ZtZi * ZtY)
    KY   = (1 - κ) .* Y .+ κ .* PZ_Y

    XKX = X_all' * KX
    β   = XKX \ (X_all' * KY)
    u   = Y .- X_all * β

    if robust
        meat   = KX' * LinearAlgebra.Diagonal(u.^2) * KX
        XKXinv = LinearAlgebra.inv(XKX)
        V      = XKXinv * meat * XKXinv'
    else
        σ² = sum(u.^2) / (n - size(X_all, 2))
        V  = σ² .* LinearAlgebra.inv(XKX)
    end
    se        = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    coefnames = vcat("_cons", string.(endog_v), string.(exog_v))
    return (; β, V, se, coefnames, n, κ)
end

"""
    stata_jive(df, y, endog, instruments, exog=Symbol[]; robust=true)

Jackknife IV (JIVE1 of Angrist, Imbens & Krueger 1999). Returns
`(; β, V, se, coefnames, n)`.
"""
function stata_jive(df, y, endog, instruments, exog = Symbol[];
                    robust::Bool = true)
    ys      = Symbol(y)
    endog_v = endog isa Union{Symbol,AbstractString} ?
              [Symbol(endog)] : [Symbol(v) for v in endog]
    inst_v  = instruments isa Union{Symbol,AbstractString} ?
              [Symbol(instruments)] : [Symbol(v) for v in instruments]
    exog_v  = isempty(exog) ? Symbol[] :
              (exog isa Union{Symbol,AbstractString} ?
                   [Symbol(exog)] : [Symbol(v) for v in exog])

    needed = unique(vcat(ys, endog_v, exog_v, inst_v))
    d = DataFrames.dropmissing(df, needed)
    n = DataFrames.nrow(d)
    Y = Float64.(_sm_rawval.(d[!, ys]))
    X_endog = hcat([Float64.(_sm_rawval.(d[!, v])) for v in endog_v]...)
    X_exog  = isempty(exog_v) ? zeros(n, 0) :
              hcat([Float64.(_sm_rawval.(d[!, v])) for v in exog_v]...)
    Z_excl  = hcat([Float64.(_sm_rawval.(d[!, v])) for v in inst_v]...)

    X_all  = hcat(ones(n), X_endog, X_exog)
    Z_full = hcat(ones(n), X_exog, Z_excl)

    ZtZi     = LinearAlgebra.inv(Z_full' * Z_full)
    X_hat_ez = Z_full * (ZtZi * (Z_full' * X_endog))
    ZZi      = Z_full * ZtZi
    h        = vec(sum(ZZi .* Z_full, dims = 2))

    X_jive_endog = (X_hat_ez .- h .* X_endog) ./ (1 .- h)
    X_jive       = hcat(ones(n), X_jive_endog, X_exog)

    XjX = X_jive' * X_all
    β   = XjX \ (X_jive' * Y)
    u   = Y .- X_all * β

    if robust
        meat = X_jive' * LinearAlgebra.Diagonal(u.^2) * X_jive
        XjXi = LinearAlgebra.inv(XjX)
        V    = XjXi * meat * XjXi'
    else
        σ² = sum(u.^2) / (n - size(X_all, 2))
        V  = σ² .* LinearAlgebra.inv(XjX)
    end
    se        = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    coefnames = vcat("_cons", string.(endog_v), string.(exog_v))
    return (; β, V, se, coefnames, n)
end

"""
    stata_estimates_table(models, labels; vars_order=nothing,
                          fmt="%9.5f", stats=String[])

Stata-style `estimates table` for side-by-side model comparison. Each
element of `models` is a NamedTuple with `β`, `se`, `coefnames`. Labels
are truncated to 9 chars in Stata style (`"twosls_def" -> "twosls_~f"`).
"""
function stata_estimates_table(models::AbstractVector, labels::AbstractVector;
                               vars_order = nothing,
                               fmt::AbstractString = "%9.5f",
                               stats::AbstractVector = String[])
    length(models) == length(labels) ||
        error("models and labels must have the same length")
    _trunc9(s) = length(s) <= 9 ? s : s[1:7] * "~" * s[end:end]
    labels_disp = _trunc9.(String.(labels))

    all_nm = String[]
    for m in models, nn in m.coefnames
        nn in all_nm || push!(all_nm, nn)
    end
    vars = vars_order === nothing ? all_nm : String.(vars_order)

    n_est = length(models)
    mfmt  = match(r"%-?(\d+)", fmt)
    col_w = mfmt === nothing ? 12 : parse(Int, mfmt.captures[1]) + 3
    var_w = 12
    total = var_w + 2 + n_est * col_w

    cell_fmt = Printf.Format(" " * String(fmt) * "  ")

    _center(s, w) = begin
        slen = length(s)
        slen >= w && return first(s, w)
        lead  = max(div(w - slen - 1, 2), 0)
        trail = w - lead - slen
        " "^lead * s * " "^trail
    end

    println("-"^total)
    hdr = lpad("Variable", var_w) * " |"
    for lbl in labels_disp
        hdr *= _center(lbl, col_w)
    end
    println(hdr)
    println("-"^var_w * "+" * "-"^(total - var_w - 1))

    blank = " "^col_w
    for v in vars
        row_b = lpad(v, var_w) * " |"
        row_s = lpad("", var_w) * " |"
        for m in models
            i = findfirst(==(v), m.coefnames)
            if i === nothing
                row_b *= blank; row_s *= blank
            else
                row_b *= Printf.format(cell_fmt, m.β[i])
                row_s *= Printf.format(cell_fmt, m.se[i])
            end
        end
        println(row_b); println(row_s)
    end

    if !isempty(stats)
        println("-"^var_w * "+" * "-"^(total - var_w - 1))
        for s in stats
            s_str = String(s)
            row = lpad(s_str, var_w) * " |"
            for m in models
                val = hasproperty(m, :stats) ? get(m.stats, s_str, nothing) : nothing
                if val === nothing
                    row *= blank
                elseif val isa Integer
                    row *= Printf.format(cell_fmt, Float64(val))
                    row = replace(row,
                        r"(?<=\s)(-?\d+)\.0+(?=\s)" => s"\1",
                        count = 1)
                else
                    row *= Printf.format(cell_fmt, val)
                end
            end
            println(row)
        end
    end
    println("-"^total)
end
