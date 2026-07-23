"""
    stata_margins(model, df, varname; kind=:dydx, level=0.95)

Stata-style `margins, dydx(<var>)` (`kind=:dydx`) or `margins, eyex(<var>) atmean`
(`kind=:eyex`) for a linear model fit on `df`.

`:dydx` — average marginal effect. Supports continuous × categorical
interactions of the form `varname + g + varname & g`:

    AME = β_varname + Σ_k (n_k / N) · β_{varname × g = k}

`:eyex` — elasticity at the sample means:  β_varname · (x̄ / ŷ(x̄)).

Delta-method SE uses the model's own vcov. Returns a NamedTuple with the
estimate, SE, z, p, and CI.
"""
function stata_margins(model, df, varname::Symbol;
                       kind::Symbol = :dydx, level::Float64 = 0.95)
    kind === :dydx && return _margins_dydx(model, df, varname; level)
    kind === :eyex && return _margins_eyex(model, df, varname; level)
    error("stata_margins: kind must be :dydx or :eyex, got $kind")
end

function _margins_dydx(model, df, varname::Symbol; level::Float64 = 0.95)
    β   = StatsBase.coef(model)
    V   = StatsBase.vcov(model)
    nm  = string.(StatsBase.coefnames(model))
    N   = Int(StatsBase.nobs(model))
    vstr = string(varname)

    idx_main = findfirst(==(vstr), nm)
    idx_main === nothing && error("'$vstr' not found as a main effect in model")

    w = zeros(length(β))
    w[idx_main] = 1.0
    re_a = Regex("^\\Q$vstr\\E\\s*&\\s*(.+?):\\s*(.+)\$")
    re_b = Regex("^(.+?):\\s*(.+?)\\s*&\\s*\\Q$vstr\\E\$")
    for (i, n) in enumerate(nm)
        m = match(re_a, n); m === nothing && (m = match(re_b, n))
        m === nothing && continue
        factor_col = Symbol(String(m.captures[1]))
        level_str  = String(m.captures[2])
        hasproperty(df, factor_col) || continue
        col = df[!, factor_col]
        lev = tryparse(Float64, level_str)
        n_k = lev === nothing ?
              sum(!ismissing(v) && string(_sm_rawval(v)) == level_str for v in col) :
              sum(!ismissing(v) && _sm_rawval(v) == lev for v in col)
        w[i] = n_k / N
    end

    ame = LinearAlgebra.dot(w, β)
    Vw  = V * w
    all(isfinite, Vw) || @warn "vcov has non-finite entries on relevant rows; SE may be NaN."

    se   = sqrt(max(LinearAlgebra.dot(w, Vw), 0.0))
    z    = se > 0 ? ame / se : NaN
    pval = se > 0 ?
           2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z))) : NaN
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    lo, hi = ame - crit * se, ame + crit * se

    nobs_str = replace(string(N), r"(\d)(?=(\d{3})+$)" => s"\1,")
    Printf.@printf("\nAverage marginal effects%40sNumber of obs = %s\n",
                   "", nobs_str)
    Printf.@printf("Expression: Linear prediction, predict()\n")
    Printf.@printf("dy/dx wrt:  %s\n\n", vstr)
    println("-"^78)
    Printf.@printf("             |            Delta-method\n")
    Printf.@printf("             |      dy/dx   std. err.      z    P>|z|     [%d%% conf. interval]\n",
                   round(Int, 100 * level))
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %10.6f  %10.6f   %5.2f   %5.3f    %10.5f  %10.5f\n",
                   vstr, ame, se, z, pval, lo, hi)
    println("-"^78)

    return (ame = ame, se = se, z = z, p = pval, ci = (lo, hi))
end

function _xbar_from_coefnames(nm, df)
    xbar = zeros(length(nm))
    N = DataFrames.nrow(df)
    for (i, n) in enumerate(nm)
        if n == "(Intercept)"
            xbar[i] = 1.0
            continue
        end
        sym = Symbol(n)
        if hasproperty(df, sym)
            col = df[!, sym]
            xbar[i] = Statistics.mean(Float64(_sm_rawval(v)) for v in skipmissing(col))
            continue
        end
        m = match(r"^(.+?):\s*(.+)$", n)
        if m !== nothing
            col_sym = Symbol(String(m.captures[1]))
            lev_str = String(m.captures[2])
            if hasproperty(df, col_sym)
                col = df[!, col_sym]
                lev = tryparse(Float64, lev_str)
                xbar[i] = lev === nothing ?
                    count(!ismissing(v) && string(_sm_rawval(v)) == lev_str for v in col) / N :
                    count(!ismissing(v) && _sm_rawval(v) == lev for v in col) / N
                continue
            end
        end
        @warn "stata_margins: could not resolve mean for coefficient '$n'; using 0."
    end
    return xbar
end

function _margins_eyex(model, df, varname::Symbol; level::Float64 = 0.95)
    β   = StatsBase.coef(model)
    V   = StatsBase.vcov(model)
    nm  = string.(StatsBase.coefnames(model))
    vstr = string(varname)

    idx = findfirst(==(vstr), nm)
    idx === nothing && error("'$vstr' not found as a main effect in model")

    if any(occursin("$vstr &", n) || occursin("& $vstr", n) for n in nm)
        @warn "Interactions involving '$vstr' detected — stata_margins(:eyex) ignores them."
    end

    xbar = _xbar_from_coefnames(nm, df)
    yhat = LinearAlgebra.dot(xbar, β)
    x̄    = Statistics.mean(skipmissing(df[!, varname]))
    dydx = β[idx]
    ela  = dydx * x̄ / yhat

    w = zeros(length(β)); w[idx] = 1.0
    g = (x̄ / yhat) .* (w .- (dydx / yhat) .* xbar)
    Vg = V * g
    se = sqrt(max(LinearAlgebra.dot(g, Vg), 0.0))
    z  = se > 0 ? ela / se : NaN
    pval = se > 0 ?
           2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z))) : NaN
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    lo, hi = ela - crit * se, ela + crit * se

    Printf.@printf("margins, eyex(%s) atmean\n", vstr)
    Printf.@printf("  ey/ex = %.6f   Std. err. = %.6f   z = %.3f   P>|z| = %.4f\n",
                   ela, se, z, pval)
    Printf.@printf("  %d%% CI = [%.6f, %.6f]\n",
                   round(Int, 100 * level), lo, hi)
    return (elasticity = ela, se = se, z = z, p = pval, ci = (lo, hi))
end
