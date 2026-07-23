"""
    _design_matrix_from_df(nm, df)

Reconstruct an `n × length(nm)` design matrix by interpreting each
coefficient name. Handles `(Intercept)`, continuous main effects, and
categorical dummies of the form `"col: level"`. Unknown terms contribute
a zero column (with a warning).
"""
function _design_matrix_from_df(nm::Vector{String}, df)
    n = DataFrames.nrow(df)
    X = zeros(n, length(nm))
    for (i, name) in enumerate(nm)
        if name == "(Intercept)"
            X[:, i] .= 1.0
            continue
        end
        sym = Symbol(name)
        if hasproperty(df, sym)
            col = df[!, sym]
            X[:, i] .= [ismissing(v) ? 0.0 : Float64(_sm_rawval(v)) for v in col]
            continue
        end
        m = match(r"^(.+?):\s*(.+)$", name)
        if m !== nothing
            col_sym = Symbol(String(m.captures[1]))
            lev_str = String(m.captures[2])
            if hasproperty(df, col_sym)
                col = df[!, col_sym]
                lev = tryparse(Float64, lev_str)
                X[:, i] .= lev === nothing ?
                    [!ismissing(v) && string(_sm_rawval(v)) == lev_str ? 1.0 : 0.0 for v in col] :
                    [!ismissing(v) && _sm_rawval(v) == lev ? 1.0 : 0.0 for v in col]
                continue
            end
        end
        @warn "estat: could not reconstruct design column for '$name'; using zeros."
    end
    return X
end

"""
    estat_imtest(model, df)

Stata-style `estat imtest` — Cameron & Trivedi's decomposition of White's
information matrix test. Prints the four chi² components (Heteroskedasticity,
Skewness, Kurtosis, Total) and returns them as a NamedTuple.

Follows Cameron & Trivedi (2005) *Microeconometrics* §8.6. Small numerical
differences from Stata's `estat imtest` are expected because of different
rank-adjustment and redundancy-handling conventions.
"""
function estat_imtest(model, df)
    nm = string.(StatsBase.coefnames(model))
    X  = _design_matrix_from_df(nm, df)
    β  = StatsBase.coef(model)

    yname = try
        Symbol(string(FixedEffectModels.formula(model).lhs))
    catch
        Symbol(string(StatsModels.formula(model).lhs))
    end
    ycol = df[!, yname]

    keep = .!ismissing.(ycol) .& .!any(isnan, X; dims = 2)[:]
    X    = X[keep, :]
    y    = Float64.(collect(skipmissing(ycol[keep])))
    e    = y .- X * β
    n, k = size(X)
    s2   = sum(abs2, e) / n

    idx_nz = [i for i in 1:k if nm[i] != "(Intercept)"]
    p  = length(idx_nz)
    Xp = X[:, idx_nz]

    function nR2_unc(dep, Z)
        b   = Z \ dep
        res = dep .- Z * b
        uss = sum(abs2, dep)
        rss = sum(abs2, res)
        return uss > 0 ? n * (1 - rss / uss) : 0.0
    end

    Xp_c = Xp .- Statistics.mean(Xp, dims = 1)
    het_cols = Vector{Vector{Float64}}()
    for a in 1:p, b in a:p
        col = Xp_c[:, a] .* Xp_c[:, b]
        if a == b && abs(Statistics.cor(col, Xp[:, a])) > 0.9999
            continue
        end
        push!(het_cols, col)
    end
    Z_het = isempty(het_cols) ? hcat(ones(n), Xp) :
            hcat(ones(n), Xp, reduce(hcat, het_cols))
    dep_h = e .^ 2 .- s2
    H     = nR2_unc(dep_h, Z_het)
    df_H  = size(Z_het, 2) - 1

    dep_s = e .^ 3 .- 3.0 .* s2 .* e
    Z_skew = hcat(ones(n), Xp)
    S     = nR2_unc(dep_s, Z_skew)
    df_S  = p

    dep_k = e .^ 4 .- 6.0 .* s2 .* e .^ 2 .+ 3.0 .* s2^2
    Z_kurt = ones(n, 1)
    K     = nR2_unc(dep_k, Z_kurt)
    df_K  = 1

    IM    = H + S + K
    df_IM = df_H + df_S + df_K

    p_H  = 1 - Distributions.cdf(Distributions.Chisq(max(df_H, 1)),  H)
    p_S  = 1 - Distributions.cdf(Distributions.Chisq(max(df_S, 1)),  S)
    p_K  = 1 - Distributions.cdf(Distributions.Chisq(df_K),           K)
    p_IM = 1 - Distributions.cdf(Distributions.Chisq(max(df_IM, 1)), IM)

    println("\nCameron & Trivedi's decomposition of IM-test\n")
    println("-"^50)
    Printf.@printf("%20s | %10s %5s %10s\n", "Source", "chi2", "df", "p")
    println("-"^21, "+", "-"^28)
    Printf.@printf("%20s | %10.2f %5d %10.4f\n", "Heteroskedasticity", H, df_H, p_H)
    Printf.@printf("%20s | %10.2f %5d %10.4f\n", "Skewness", S, df_S, p_S)
    Printf.@printf("%20s | %10.2f %5d %10.4f\n", "Kurtosis", K, df_K, p_K)
    println("-"^21, "+", "-"^28)
    Printf.@printf("%20s | %10.2f %5d %10.4f\n", "Total", IM, df_IM, p_IM)
    println("-"^50)

    return (H     = (chi2 = H,  df = df_H,  p = p_H),
            S     = (chi2 = S,  df = df_S,  p = p_S),
            K     = (chi2 = K,  df = df_K,  p = p_K),
            total = (chi2 = IM, df = df_IM, p = p_IM))
end

"""
    estat_hettest(model, df; vars=nothing, iid=false, mtest=false)

Breusch-Pagan / Cook-Weisberg test for heteroskedasticity. Replicates
Stata's `estat hettest [varlist] [, iid] [, mtest]`.

- `vars=nothing`  : test against fitted values (1 df).
- `vars=[…]`      : test against the listed variables (joint + individual).
- `iid=true`      : NR² form (regress u²/σ² on z, LM = n·R²), matching
                    Stata's `estat hettest, iid`.
- `iid=false`     : score form (Koenker, 1981) — Stata's default.
- `mtest=true`    : also report per-variable Bonferroni-style unadjusted
                    p-values.

Returns `(chi2, df, p)`.
"""
function estat_hettest(model, df;
                       vars::Union{Nothing, Vector{Symbol}} = nothing,
                       iid::Bool = false,
                       mtest::Bool = false)
    nm = StatsBase.coefnames(model)
    n  = Int(StatsBase.nobs(model))

    needed = Symbol[]
    for name in nm
        name == "(Intercept)" && continue
        s = Symbol(strip(split(name, ":")[1]))
        hasproperty(df, s) && push!(needed, s)
    end
    resp = try
        Symbol(string(FixedEffectModels.formula(model).lhs))
    catch
        Symbol(string(StatsModels.formula(model).lhs))
    end
    push!(needed, resp)
    unique!(needed)
    dfc = DataFrames.dropmissing(df, needed)

    yhat = FixedEffectModels.predict(model, dfc)
    uhat = Float64.(dfc[!, resp]) .- yhat
    sigma2 = sum(uhat .^ 2) / n

    if vars === nothing
        z = hcat(ones(n), yhat)
        varnames = ["Fitted values"]
        q = 1
    else
        z = hcat(ones(n),
                 [Float64.([_sm_rawval(x) for x in dfc[!, v]]) for v in vars]...)
        varnames = string.(vars)
        q = length(vars)
    end

    function _bp_chi2(dep, z, is_iid)
        b = z \ dep
        fitted = z * b
        if is_iid
            ss_tot = sum((dep .- Statistics.mean(dep)) .^ 2)
            ss_res = sum((dep .- fitted) .^ 2)
            return ss_tot > 0 ? n * (1 - ss_res / ss_tot) : 0.0
        else
            ess = sum((fitted .- Statistics.mean(dep)) .^ 2)
            return ess / (2 * sigma2^2)
        end
    end

    chi2 = _bp_chi2(uhat .^ 2, z, iid)
    pval = 1 - Distributions.cdf(Distributions.Chisq(q), chi2)

    form = iid ? "i.i.d." : "Normal"
    if mtest && vars !== nothing && length(vars) > 1
        println("\nBreusch-Pagan / Cook-Weisberg test for heteroskedasticity")
        Printf.@printf("Assumption: %s error terms\n", form)
        println("H0: Constant variance\n")
        println("-"^38)
        Printf.@printf("    Variable |      chi2   df        p\n")
        println("-"^13, "+", "-"^24)
        dep_sq = uhat .^ 2
        for v in vars
            zj = hcat(ones(n),
                      Float64.([_sm_rawval(x) for x in dfc[!, v]]))
            chi2_j = _bp_chi2(dep_sq, zj, iid)
            p_j = 1 - Distributions.cdf(Distributions.Chisq(1), chi2_j)
            Printf.@printf("%12s | %9.2f    1   %.4f*\n", v, chi2_j, p_j)
        end
        println("-"^13, "+", "-"^24)
        Printf.@printf("Simultaneous | %9.2f    %d   %.4f\n", chi2, q, pval)
        println("-"^38)
        println("* Unadjusted p-values")
    else
        var_label = vars === nothing ? "Fitted values of $(resp)" :
                                       join(varnames, ", ")
        println("\nBreusch-Pagan / Cook-Weisberg test for heteroskedasticity")
        Printf.@printf("Assumption: %s error terms\n", form)
        println("Variable: $(var_label)")
        println("\nH0: Constant variance\n")
        Printf.@printf("    chi2(%d) = %6.2f\n", q, chi2)
        Printf.@printf("Prob > chi2 = %.4f\n", pval)
    end

    return (chi2 = chi2, df = q, p = pval)
end
