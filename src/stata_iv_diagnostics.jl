# --------------------------------------------------------------------------
# IV post-estimation diagnostics (Cameron & Trivedi ch. 6.3.6–6.4.3):
#   estat_endogenous  — Durbin-Wu-Hausman test of endogeneity
#   estat_overid      — Hansen's J test of overidentifying restrictions
#   estat_firststage  — first-stage summary + Shea's partial R² +
#                       Stock-Yogo critical values
# --------------------------------------------------------------------------

"""
    estat_endogenous(df, y, endog, instruments, exog=Symbol[]; robust=true)

Stata-style `estat endogenous` after `ivregress 2sls`. Wooldridge
control-function form of the Durbin-Wu-Hausman test: regress `y` on
`[endog, v̂, exog]` (with `v̂` = first-stage residuals) and jointly test
the v̂ coefficients. Reports both the χ² and F forms.
"""
function estat_endogenous(df, y, endog, instruments, exog = Symbol[];
                          robust::Bool = true)
    ys      = Symbol(y)
    endog_v = endog isa Union{Symbol,AbstractString} ?
              [Symbol(endog)] : [Symbol(v) for v in endog]
    iv_v    = instruments isa Union{Symbol,AbstractString} ?
              [Symbol(instruments)] : [Symbol(v) for v in instruments]
    exog_v  = isempty(exog) ? Symbol[] :
              (exog isa Union{Symbol,AbstractString} ?
                  [Symbol(exog)] : [Symbol(v) for v in exog])

    needed = unique(vcat(ys, endog_v, exog_v, iv_v))
    d = DataFrames.dropmissing(df, needed)

    resid_names = Symbol[]
    for ev in endog_v
        f_fs = term(ev) ~ sum(term.(vcat(exog_v, iv_v)))
        m_fs = FixedEffectModels.reg(d, f_fs)
        vn   = Symbol("_v_", ev)
        d[!, vn] = Float64.(_sm_rawval.(d[!, ev])) .-
                   FixedEffectModels.predict(m_fs, d)
        push!(resid_names, vn)
    end

    vcov = robust ? FixedEffectModels.Vcov.robust() :
                    FixedEffectModels.Vcov.simple()
    f_aux = term(ys) ~ sum(term.(vcat(endog_v, resid_names, exog_v)))
    m_aux = FixedEffectModels.reg(d, f_aux, vcov)

    β  = StatsBase.coef(m_aux)
    V  = StatsBase.vcov(m_aux)
    nm = string.(StatsBase.coefnames(m_aux))
    idx = [findfirst(==(string(v)), nm) for v in resid_names]
    β_r = β[idx]; V_r = V[idx, idx]
    q   = length(idx)
    W   = β_r' * LinearAlgebra.inv(V_r) * β_r
    n   = Int(StatsBase.nobs(m_aux)); k = length(β)
    df2 = n - k
    F   = W / q
    p_chi = 1 - Distributions.cdf(Distributions.Chisq(q), W)
    p_F   = 1 - Distributions.cdf(Distributions.FDist(q, df2), F)

    prefix = robust ? "Robust" : ""
    println("  Tests of endogeneity")
    println("  H0: Variables are exogenous\n")
    Printf.@printf("    %s score chi2(%d)           = %9.5f  (p = %.4f)\n",
                   prefix, q, W, p_chi)
    Printf.@printf("    %s regression F(%d,%d)%s  = %9.5f  (p = %.4f)\n",
                   prefix, q, df2, " "^max(0, 4 - length(string(df2))),
                   F, p_F)
    return (; chi2 = W, df1 = q, p_chi, F, df2, p_F,
              endog_vars = endog_v, instruments = iv_v, exog_vars = exog_v)
end

"""
    estat_overid(df, y, endog, instruments, exog=Symbol[]; cluster=nothing)

Stata-style `estat overid` after `ivregress gmm ..., wmatrix(robust)`.
Two-step efficient GMM Hansen J: `(Z'û)' Ω̂⁻¹ (Z'û) ~ χ²(L − K)`.
"""
function estat_overid(df, y, endog, instruments, exog = Symbol[];
                      cluster::Union{Symbol,Nothing} = nothing)
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

    W0 = LinearAlgebra.inv(Z' * Z)
    XZ = X' * Z
    β0 = (XZ * W0 * XZ') \ (XZ * W0 * (Z' * y_vec))
    u0 = y_vec .- X * β0

    m = size(Z, 2)
    Ω = zeros(m, m)
    if cl_vec === nothing
        for i in 1:n
            zi = @view Z[i, :]
            Ω .+= (u0[i]^2) .* (zi * zi')
        end
    else
        for g in unique(cl_vec)
            sel = cl_vec .== g
            zu  = Z[sel, :]' * u0[sel]
            Ω  .+= zu * zu'
        end
    end

    W = LinearAlgebra.inv(Ω)
    β = (XZ * W * XZ') \ (XZ * W * (Z' * y_vec))
    u = y_vec .- X * β

    gvec  = Z' * u
    J     = gvec' * W * gvec
    df_J  = size(Z, 2) - size(X, 2)
    p_val = df_J > 0 ? 1 - Distributions.cdf(Distributions.Chisq(df_J), J) : NaN

    println("Test of overidentifying restriction:\n")
    if df_J <= 0
        println("  Model is just-identified (df = 0); J test not applicable.")
    else
        Printf.@printf("  Hansen's J chi2(%d) = %.5f (p = %.4f)\n",
                       df_J, J, p_val)
    end
    return (; J, df = df_J, p = p_val, n)
end

"""
    estat_firststage(df, y, endog, instruments, exog=Symbol[];
                     all=false, robust=true)

Stata-style `estat firststage[, forcenonrobust all]`. Prints the
first-stage regression summary (with Robust F on excluded instruments),
Shea's partial R² (when `all=true`), the minimum eigenvalue statistic,
and the Stock-Yogo (2005) critical values table for
`(#endog = 1, #excluded IV ∈ 1..5)`.
"""
function estat_firststage(df, y, endog, instruments, exog = Symbol[];
                          all::Bool = false, robust::Bool = true)
    ys      = Symbol(y)
    endog_v = endog isa Union{Symbol,AbstractString} ?
              [Symbol(endog)] : [Symbol(v) for v in endog]
    inst_v  = instruments isa Union{Symbol,AbstractString} ?
              [Symbol(instruments)] : [Symbol(v) for v in instruments]
    exog_v  = isempty(exog) ? Symbol[] :
              (exog isa Union{Symbol,AbstractString} ?
                   [Symbol(exog)] : [Symbol(v) for v in exog])

    needed  = unique(vcat(ys, endog_v, exog_v, inst_v))
    d       = DataFrames.dropmissing(df, needed)
    n       = DataFrames.nrow(d)
    n_endog = length(endog_v)
    L       = length(inst_v)

    fs_rows      = NTuple{8, Any}[]
    shea_rows    = NTuple{3, Any}[]
    min_eig_vals = Float64[]

    for ev in endog_v
        f_full = term(ev) ~ sum(term.(vcat(exog_v, inst_v)))
        m_cls  = FixedEffectModels.reg(d, f_full)
        m_rob  = robust ? FixedEffectModels.reg(d, f_full,
                                                FixedEffectModels.Vcov.robust()) :
                          m_cls
        RSS_full = StatsBase.deviance(m_cls)
        RSS_restr = if isempty(exog_v)
            y_ev = Float64.(_sm_rawval.(d[!, ev]))
            sum((y_ev .- Statistics.mean(y_ev)).^2)
        else
            f_restr = term(ev) ~ sum(term.(exog_v))
            StatsBase.deviance(FixedEffectModels.reg(d, f_restr))
        end
        R²_full    = StatsBase.r2(m_cls)
        adjR²_full = StatsBase.adjr2(m_cls)
        partial_R² = (RSS_restr - RSS_full) / RSS_restr

        β      = StatsBase.coef(m_cls)
        nmvec  = string.(StatsBase.coefnames(m_cls))
        iv_idx = [findfirst(==(string(v)), nmvec) for v in inst_v]
        β_iv   = β[iv_idx]
        df_den = Int(StatsBase.dof_residual(m_cls))

        V_main = StatsBase.vcov(m_rob)
        V_iv_m = V_main[iv_idx, iv_idx]
        F_main = (β_iv' * LinearAlgebra.inv(V_iv_m) * β_iv) / L
        p_main = 1 - Distributions.cdf(Distributions.FDist(L, df_den), F_main)
        push!(fs_rows, (string(ev), R²_full, adjR²_full, partial_R²,
                         F_main, p_main, L, df_den))

        k_total  = length(β)
        shea_adj = 1 - (1 - partial_R²) * (n - 1) / (n - k_total)
        push!(shea_rows, (string(ev), partial_R², shea_adj))

        V_cls  = StatsBase.vcov(m_cls)
        V_iv_c = V_cls[iv_idx, iv_idx]
        F_cls  = (β_iv' * LinearAlgebra.inv(V_iv_c) * β_iv) / L
        push!(min_eig_vals, F_cls)
    end

    # Block 1: first-stage summary
    println()
    println("  First-stage regression summary statistics")
    println("  " * "-"^74)
    label_robust = robust ? "Robust" : "      "
    println("               |            Adjusted      Partial       $(label_robust)")
    Printf.@printf("      Variable |   R-sq.       R-sq.        R-sq.    F(%d,%d)   Prob > F\n",
                   fs_rows[1][7], fs_rows[1][8])
    println("  -------------+" * "-"^60)
    for r in fs_rows
        Printf.@printf("  %12s | %6.4f      %6.4f       %6.4f    %10.4f    %.4f\n",
                       r[1], r[2], r[3], r[4], r[5], r[6])
    end
    println("  " * "-"^74)
    println()

    # Block 2: Shea's partial R²
    if all
        println()
        println("  Shea's partial R-squared")
        println("  " * "-"^50)
        println("               |     Shea's             Shea's")
        println("      Variable |  partial R-sq.   adj. partial R-sq.")
        println("  -------------+" * "-"^36)
        for r in shea_rows
            Printf.@printf("  %12s |     %6.4f             %6.4f\n",
                           r[1], r[2], r[3])
        end
        println("  " * "-"^50)
        println()
    end

    # Block 3: Minimum eigenvalue + Stock-Yogo table
    println()
    Printf.@printf("  Minimum eigenvalue statistic = %.2f\n\n", min_eig_vals[1])

    # Stock-Yogo (2005) critical values for n_endog = 1
    sy_2sls_size = Dict(
        1 => (16.38,  8.96,  6.66,  5.53),
        2 => (19.93, 11.59,  8.75,  7.25),
        3 => (22.30, 12.83,  9.54,  7.80),
        4 => (24.58, 13.96, 10.26,  8.31),
        5 => (26.87, 15.09, 10.98,  8.82),
    )
    sy_liml_size = Dict(
        1 => (16.38, 8.96, 6.66, 5.53),
        2 => ( 8.68, 5.33, 4.42, 3.92),
        3 => ( 6.46, 4.36, 3.69, 3.32),
        4 => ( 5.44, 3.87, 3.30, 2.98),
        5 => ( 4.84, 3.56, 3.05, 2.77),
    )
    sy_2sls_bias = Dict(
        3 => (13.91, 9.08, 6.46, 5.39),
        4 => (16.85, 10.27, 6.71, 5.34),
        5 => (18.37, 10.83, 6.77, 5.25),
    )

    Printf.@printf("  Critical Values                      # of endogenous regressors: %4d\n", n_endog)
    Printf.@printf("  H0: Instruments are weak             # of excluded instruments:  %4d\n", L)
    println("  " * "-"^69)
    println("                                     |    5%     10%     20%     30%")
    if n_endog == 1 && haskey(sy_2sls_bias, L)
        b = sy_2sls_bias[L]
        Printf.@printf("  2SLS relative bias                 |  %5.2f   %5.2f   %5.2f   %5.2f\n",
                       b[1], b[2], b[3], b[4])
    else
        println("  2SLS relative bias                 |         (not available)")
    end
    println("  -----------------------------------+" * "-"^33)
    println("                                     |   10%     15%     20%     25%")
    if n_endog == 1 && haskey(sy_2sls_size, L)
        v = sy_2sls_size[L]
        Printf.@printf("  2SLS size of nominal 5%% Wald test  |  %5.2f   %5.2f   %5.2f   %5.2f\n",
                       v[1], v[2], v[3], v[4])
    else
        println("  2SLS size of nominal 5% Wald test  |         (not available)")
    end
    if n_endog == 1 && haskey(sy_liml_size, L)
        v = sy_liml_size[L]
        Printf.@printf("  LIML size of nominal 5%% Wald test  |  %5.2f   %5.2f   %5.2f   %5.2f\n",
                       v[1], v[2], v[3], v[4])
    else
        println("  LIML size of nominal 5% Wald test  |         (not available)")
    end
    println("  " * "-"^69)
    return (; fs_rows, shea_rows, min_eig = min_eig_vals, n, n_endog, L)
end
