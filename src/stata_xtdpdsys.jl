# ============================================================================
# stata_xtdpdsys.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_xtdpdsys(df, y, idvar, timevar; lags, xvars, pre, endog, maxldep, ma_order, twostep, artests)

Stata-style `xtdpdsys` — Blundell-Bond (1998) system GMM (pedagogical). Stacks
the Arellano-Bond differenced moment conditions with the Blundell-Bond level
moment conditions; two-step uses the Windmeijer (2005) finite-sample correction.
"""
function stata_xtdpdsys(df, y::Symbol, idvar::Symbol, timevar::Symbol;
                  lags::Int=1,
                  xvars::Vector{Symbol}=Symbol[],
                  pre::Vector{Tuple{Symbol,Int,Int}}=Tuple{Symbol,Int,Int}[],
                  endog::Vector{Tuple{Symbol,Int,Int}}=Tuple{Symbol,Int,Int}[],
                  maxldep::Int=0,
                  ma_order::Int=0,
                  twostep::Bool=false,
                  artests::Int=2)

    all_vars = unique(vcat([y, idvar, timevar], xvars,
                           [v for (v,_,_) in pre], [v for (v,_,_) in endog]))
    d = DataFrames.dropmissing(DataFrames.sort(df, [idvar, timevar]), all_vars)
    grouped = DataFrames.groupby(d, idvar)
    N_ind = length(grouped)
    T_max = maximum(DataFrames.nrow(g) for g in grouped)

    t_start_diff = lags + 2
    t_start_lev  = lags + 1
    n_per_diff = T_max - t_start_diff + 1
    n_per_lev  = T_max - t_start_lev + 1

    diff_gmm = [(y, lags + ma_order, maxldep > 0 ? maxldep : 999)]
    for (v, a, b) in pre;    push!(diff_gmm, (v, a + 1, b)); end
    for (v, a, b) in endog;  push!(diff_gmm, (v, a + 2, b)); end

    level_gmm_vars = vcat([y], [v for (v,_,_) in pre], [v for (v,_,_) in endog])
    n_lev_vars = length(level_gmm_vars)

    diff_period_info = Vector{Vector{Tuple{Int,Int}}}(undef, n_per_diff)
    col = 1
    for (pidx, t) in enumerate(t_start_diff:T_max)
        info = Tuple{Int,Int}[]
        for (_, s, depth) in diff_gmm
            n_inst = max(0, min(depth, t - s))
            push!(info, (col, n_inst))
            col += n_inst
        end
        diff_period_info[pidx] = info
    end
    n_diff_gmm = col - 1
    n_diff_std = length(xvars)
    n_diff_z   = n_diff_gmm + n_diff_std

    n_lev_gmm = n_per_lev * n_lev_vars
    n_lev_std = 1
    n_lev_z   = n_lev_gmm + n_lev_std

    n_z = n_diff_z + n_lev_z

    n_pre_regs   = sum(a + 1 for (_,a,_) in pre;   init=0)
    n_endog_regs = sum(a + 1 for (_,a,_) in endog; init=0)
    k_reg_no_cons = lags + n_pre_regs + n_endog_regs + length(xvars)
    k_reg = k_reg_no_cons + 1

    diff_y_all = Float64[]; diff_X_all = Matrix{Float64}(undef, 0, k_reg)
    diff_Z_all = Matrix{Float64}(undef, 0, n_z)
    lev_y_all  = Float64[]; lev_X_all  = Matrix{Float64}(undef, 0, k_reg)
    lev_Z_all  = Matrix{Float64}(undef, 0, n_z)

    T_diff_per_panel = Int[]; T_lev_per_panel = Int[]

    panel_Z = Vector{Matrix{Float64}}()
    panel_X = Vector{Matrix{Float64}}()
    panel_y = Vector{Vector{Float64}}()
    panel_t_starts = Vector{Tuple{Int,Int}}()

    for g in grouped
        T_i = DataFrames.nrow(g)
        y_v = Float64.([_c9_rawval(v) for v in g[!, y]])
        pre_vecs   = [Float64.([_c9_rawval(x) for x in g[!, v]]) for (v,_,_) in pre]
        endog_vecs = [Float64.([_c9_rawval(x) for x in g[!, v]]) for (v,_,_) in endog]
        x_vecs     = [Float64.([_c9_rawval(x) for x in g[!, v]]) for v in xvars]
        gmm_vecs   = vcat([y_v], pre_vecs, endog_vecs)

        n_diff_i = max(0, T_i - t_start_diff + 1)
        push!(T_diff_per_panel, n_diff_i)
        dZ_i = zeros(0, n_z); dX_i = zeros(0, k_reg); dy_i = Float64[]
        if n_diff_i > 0
            dy_i = zeros(n_diff_i)
            dX_i = zeros(n_diff_i, k_reg)
            dZ_i = zeros(n_diff_i, n_z)
            for (row, t) in enumerate(t_start_diff:T_i)
                dy_i[row] = y_v[t] - y_v[t-1]
                c = 0
                for k in 1:lags
                    c += 1
                    dX_i[row, c] = y_v[t-k] - y_v[t-k-1]
                end
                for (pi, (_, a, _)) in enumerate(pre)
                    for k in 0:a
                        c += 1
                        dX_i[row, c] = pre_vecs[pi][t-k] - pre_vecs[pi][t-k-1]
                    end
                end
                for (ei, (_, a, _)) in enumerate(endog)
                    for k in 0:a
                        c += 1
                        dX_i[row, c] = endog_vecs[ei][t-k] - endog_vecs[ei][t-k-1]
                    end
                end
                for xi in eachindex(xvars)
                    c += 1
                    dX_i[row, c] = x_vecs[xi][t] - x_vecs[xi][t-1]
                end

                pidx = t - t_start_diff + 1
                for (gi, (_, s, _)) in enumerate(diff_gmm)
                    cs, n_inst = diff_period_info[pidx][gi]
                    for j in 1:n_inst
                        idx_src = t - s - n_inst + j
                        dZ_i[row, cs + j - 1] = (idx_src >= 1) ? gmm_vecs[gi][idx_src] : 0.0
                    end
                end
                for xi in eachindex(xvars)
                    dZ_i[row, n_diff_gmm + xi] = x_vecs[xi][t] - x_vecs[xi][t-1]
                end
            end
            diff_y_all = vcat(diff_y_all, dy_i)
            diff_X_all = vcat(diff_X_all, dX_i)
            diff_Z_all = vcat(diff_Z_all, dZ_i)
        end

        n_lev_i = max(0, T_i - t_start_lev + 1)
        push!(T_lev_per_panel, n_lev_i)
        lZ_i = zeros(0, n_z); lX_i = zeros(0, k_reg); ly_i = Float64[]
        if n_lev_i > 0
            ly_i = zeros(n_lev_i)
            lX_i = zeros(n_lev_i, k_reg)
            lZ_i = zeros(n_lev_i, n_z)
            for (row, t) in enumerate(t_start_lev:T_i)
                ly_i[row] = y_v[t]
                c = 0
                for k in 1:lags
                    c += 1; lX_i[row, c] = y_v[t-k]
                end
                for (pi, (_, a, _)) in enumerate(pre)
                    for k in 0:a; c += 1; lX_i[row, c] = pre_vecs[pi][t-k]; end
                end
                for (ei, (_, a, _)) in enumerate(endog)
                    for k in 0:a; c += 1; lX_i[row, c] = endog_vecs[ei][t-k]; end
                end
                for xi in eachindex(xvars)
                    c += 1; lX_i[row, c] = x_vecs[xi][t]
                end
                lX_i[row, k_reg] = 1.0

                pidx_l = t - t_start_lev + 1
                for (gi, vec) in enumerate(gmm_vecs)
                    col_in_lev = (pidx_l - 1) * n_lev_vars + gi
                    col_global = n_diff_z + col_in_lev
                    if gi == 1
                        lag_off = ma_order + 1
                        t_min  = lag_off + 2
                        if t >= t_min
                            lZ_i[row, col_global] = vec[t - lag_off] - vec[t - lag_off - 1]
                        end
                    else
                        if t >= 3
                            lZ_i[row, col_global] = vec[t-1] - vec[t-2]
                        end
                    end
                end
                lZ_i[row, n_diff_z + n_lev_gmm + 1] = 1.0
            end
            lev_y_all  = vcat(lev_y_all, ly_i)
            lev_X_all  = vcat(lev_X_all, lX_i)
            lev_Z_all  = vcat(lev_Z_all, lZ_i)
        end

        if n_diff_i > 0 || n_lev_i > 0
            Z_i = vcat(n_diff_i > 0 ? dZ_i : zeros(0, n_z),
                       n_lev_i  > 0 ? lZ_i : zeros(0, n_z))
            X_i = vcat(n_diff_i > 0 ? dX_i : zeros(0, k_reg),
                       n_lev_i  > 0 ? lX_i : zeros(0, k_reg))
            y_i = vcat(n_diff_i > 0 ? dy_i : Float64[],
                       n_lev_i  > 0 ? ly_i : Float64[])
            push!(panel_Z, Z_i)
            push!(panel_X, X_i)
            push!(panel_y, y_i)
            push!(panel_t_starts, (t_start_diff, t_start_lev))
        end
    end

    N_diff = length(diff_y_all)
    N_level = length(lev_y_all)
    N_total = N_diff + N_level
    y_sys = vcat(diff_y_all, lev_y_all)
    X_sys = vcat(diff_X_all, lev_X_all)
    Z_sys = vcat(diff_Z_all, lev_Z_all)

    nz_cols = findall(c -> any(!iszero, Z_sys[:, c]), 1:size(Z_sys, 2))
    Z_sys = Z_sys[:, nz_cols]
    panel_Z = [Z[:, nz_cols] for Z in panel_Z]
    n_zd_kept = count(c -> c <= n_diff_z, nz_cols)
    n_zl_kept = length(nz_cols) - n_zd_kept

    function build_H_i(t_diff_range, t_level_range)
        nd, nl = length(t_diff_range), length(t_level_range)
        H = zeros(nd + nl, nd + nl)
        for i in 1:nd
            H[i, i] = 2.0
            if i < nd; H[i, i+1] = -1.0; end
            if i > 1;  H[i, i-1] = -1.0; end
        end
        for i in 1:nl
            H[nd + i, nd + i] = 1.0
        end
        for (i, t) in enumerate(t_diff_range)
            for (j, s) in enumerate(t_level_range)
                if s == t
                    H[i, nd + j] = +1.0
                    H[nd + j, i] = +1.0
                elseif s == t - 1
                    H[i, nd + j] = -1.0
                    H[nd + j, i] = -1.0
                end
            end
        end
        return H
    end

    nz = size(Z_sys, 2)
    A1 = zeros(nz, nz)
    for (i, Z_i) in enumerate(panel_Z)
        nd_i = T_diff_per_panel[i]
        nl_i = T_lev_per_panel[i]
        t_diff = collect((t_start_diff):(t_start_diff + nd_i - 1))
        t_level = collect((t_start_lev):(t_start_lev + nl_i - 1))
        H_i = build_H_i(t_diff, t_level)
        A1 .+= Z_i' * H_i * Z_i
    end

    W1 = try
        LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A1)))
    catch
        LinearAlgebra.pinv(A1)
    end

    XZ = X_sys' * Z_sys
    Zy = Z_sys' * y_sys
    β1 = (XZ * W1 * XZ') \ (XZ * W1 * Zy)
    u1 = y_sys .- X_sys * β1

    u1_panel = Vector{Vector{Float64}}()
    rstart_d = 0; rstart_l = N_diff
    for i in 1:length(panel_Z)
        nd_i = T_diff_per_panel[i]; nl_i = T_lev_per_panel[i]
        u_i = vcat(diff_y_all[rstart_d+1:rstart_d+nd_i] .-
                       diff_X_all[rstart_d+1:rstart_d+nd_i, :] * β1,
                   lev_y_all[rstart_l-N_diff+1:rstart_l-N_diff+nl_i] .-
                       lev_X_all[rstart_l-N_diff+1:rstart_l-N_diff+nl_i, :] * β1)
        push!(u1_panel, u_i)
        rstart_d += nd_i; rstart_l += nl_i
    end

    β_final, V_final, W_final = β1, nothing, W1
    if twostep
        Ω1 = zeros(nz, nz)
        for i in 1:length(panel_Z)
            Ω1 .+= panel_Z[i]' * (u1_panel[i] * u1_panel[i]') * panel_Z[i]
        end
        W2 = try
            LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Ω1)))
        catch
            LinearAlgebra.pinv(Ω1)
        end
        β2 = (XZ * W2 * XZ') \ (XZ * W2 * Zy)
        u2 = y_sys .- X_sys * β2

        u2_panel = Vector{Vector{Float64}}()
        rd = 0; rl = N_diff
        for i in 1:length(panel_Z)
            nd_i = T_diff_per_panel[i]; nl_i = T_lev_per_panel[i]
            u_i = vcat(diff_y_all[rd+1:rd+nd_i] .- diff_X_all[rd+1:rd+nd_i, :] * β2,
                       lev_y_all[rl-N_diff+1:rl-N_diff+nl_i] .- lev_X_all[rl-N_diff+1:rl-N_diff+nl_i, :] * β2)
            push!(u2_panel, u_i)
            rd += nd_i; rl += nl_i
        end

        SA2 = Ω1
        B1 = LinearAlgebra.inv(XZ * W1 * XZ')
        B2 = LinearAlgebra.inv(XZ * W2 * XZ')
        We = zeros(nz)
        for i in 1:length(panel_Z)
            We .+= panel_Z[i]' * u2_panel[i]
        end
        vcov_1s_robust = B1 * (XZ * W1 * SA2 * W1 * XZ') * B1

        k_reg_total = size(X_sys, 2)
        D = zeros(k_reg_total, k_reg_total)
        for k in 1:k_reg_total
            wexkw = zeros(nz, nz)
            for i in 1:length(panel_Z)
                X_ik = panel_X[i][:, k]
                u1_i = u1_panel[i]
                exk_i = -(X_ik * u1_i' + u1_i * X_ik')
                wexkw .+= panel_Z[i]' * exk_i * panel_Z[i]
            end
            D[:, k] = -B2 * (XZ * W2 * wexkw * W2) * We
        end

        V_final = B2 + D * B2 + (D * B2)' + D * vcov_1s_robust * D'
        β_final = β2;  W_final = W2
    else
        Ω1r = zeros(nz, nz)
        for i in 1:length(panel_Z)
            Ω1r .+= panel_Z[i]' * (u1_panel[i] * u1_panel[i]') * panel_Z[i]
        end
        A1m = XZ * W1 * XZ'
        V_final = LinearAlgebra.inv(A1m) * (XZ * W1 * Ω1r * W1 * XZ') * LinearAlgebra.inv(A1m)
    end

    u_final = y_sys .- X_sys * β_final
    k_all = size(X_sys, 2)

    β_full = β_final
    se_full = sqrt.(max.(LinearAlgebra.diag(V_final), 0.0))
    V_full  = V_final

    z_all = β_full ./ se_full
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_all)))

    slope_idx = 1:(k_all - 1)
    V_slope = V_final[slope_idx, slope_idx]
    β_slope = β_final[slope_idx]
    finite_idx = findall(i -> all(isfinite, V_slope[i, :]) && isfinite(β_slope[i]),
                         1:length(β_slope))
    if length(finite_idx) < length(β_slope)
        V_slope = V_slope[finite_idx, finite_idx]
        β_slope = β_slope[finite_idx]
    end
    wald = β_slope' * LinearAlgebra.pinv(V_slope) * β_slope
    wald_df = length(β_slope)
    wald_p = 1 - Distributions.cdf(Distributions.Chisq(wald_df), wald)

    colnames = String[]
    for k in 1:lags;  push!(colnames, "L$(k).$(y)");  end
    for (v, a, _) in pre
        for k in 0:a;  push!(colnames, k == 0 ? string(v) : "L$(k).$(v)");  end
    end
    for (v, a, _) in endog
        for k in 0:a;  push!(colnames, k == 0 ? string(v) : "L$(k).$(v)");  end
    end
    for v in xvars;  push!(colnames, string(v));  end
    push!(colnames, "_cons")

    n_instr = size(Z_sys, 2)

    ci_lo_full = β_full .- 1.96 .* se_full
    ci_hi_full = β_full .+ 1.96 .* se_full

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return lpad(s, w)
    end
    function commafmt(m)
        s = string(abs(m)); parts = String[]; i = length(s)
        while i >= 1
            push!(parts, s[max(1, i-2):i]); i -= 3
        end
        return (m < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    step_lbl = twostep ? "Two-step" : "One-step"
    N_sys = N_level
    T_sys = [DataFrames.nrow(g) - lags for g in grouped]
    println()
    Printf.@printf("%-48s%-18s= %10s\n",
                   "System dynamic panel-data estimation",
                   "Number of obs", commafmt(N_sys))
    Printf.@printf("Group variable: %-32s%-18s= %10d\n",
                   string(idvar), "Number of groups", N_ind)
    Printf.@printf("Time variable: %s\n", string(timevar))
    println("                                                Obs per group:")
    Printf.@printf("%48s%-5s = %10d\n", "", "min", minimum(T_sys))
    Printf.@printf("%48s%-5s = %10d\n", "", "avg", round(Int, Statistics.mean(T_sys)))
    Printf.@printf("%48s%-5s = %10d\n", "", "max", maximum(T_sys))
    println()
    Printf.@printf("Number of instruments = %6d%18sWald chi2(%d)      = %10.2f\n",
                   n_instr, "", wald_df, wald)
    Printf.@printf("%48sProb > chi2       = %10.4f\n", "", wald_p)
    println("$(step_lbl) results")
    println("-"^78)
    if twostep
        println("             |              WC-robust")
    else
        println("             |               Robust")
    end
    Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [95%% conf. interval]\n",
                   string(y))
    println("-"^13, "+", "-"^64)

    function prow(label::String, i::Int)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       label,
                       g9(β_full[i]; w=10), g9(se_full[i]; w=9),
                       Printf.@sprintf("%6.2f", z_all[i]),
                       Printf.@sprintf("%.3f", pv[i]),
                       g9(ci_lo_full[i]; w=10), g9(ci_hi_full[i]; w=10))
    end

    idx = 1
    if lags > 0
        Printf.@printf("%12s |\n", string(y))
        for k in 1:lags;  prow("L$(k).", idx);  idx += 1;  end
        println("             |")
    end
    for (v, a, _) in pre
        Printf.@printf("%12s |\n", string(v))
        for k in 0:a
            lbl = k == 0 ? "--." : "L$(k)."
            prow(lbl, idx);  idx += 1
        end
        println("             |")
    end
    for (v, a, _) in endog
        for k in 0:a
            lbl = k == 0 ? string(v) : "L$(k).$(v)"
            prow(lbl, idx);  idx += 1
        end
    end
    for v in xvars
        prow(string(v), idx);  idx += 1
    end
    prow("_cons", length(β_full))
    println("-"^78)

    y_min_lag = lags + ma_order
    maxlag_y = maxldep>0 ? string(y_min_lag + maxldep - 1) : "."
    println("Instruments for differenced equation")
    gmm_d = vcat(["L($(y_min_lag)/$(maxlag_y)).$y"],
        ["L(2/$(1+b)).$v" for (v,_,b) in endog],
        ["L(1/$(b)).L$(a>1 ? string(a) : "").$v" for (v,a,b) in pre])
    println("        GMM-type: ", join(gmm_d, " "))
    !isempty(xvars) && println("        Standard: ", join(["D.$v" for v in xvars], " "))
    println("Instruments for level equation")
    y_lev_pref = ma_order == 0 ? "LD" : "L$(ma_order+1)D"
    gmm_l = vcat(["$(y_lev_pref).$y"],
                 ["LD.$v" for (v,_,_) in pre],
                 ["LD.$v" for (v,_,_) in endog])
    println("        GMM-type: ", join(gmm_l, " "))
    println("        Standard: _cons")

    return (; β=β_full, V=V_full, coefnames=colnames, residuals=u_final,
              Zmat=Z_sys, Xmat=X_sys, y=y_sys, twostep, W=W_final,
              T_diff_per_panel, T_lev_per_panel, N_obs=N_diff, N_ind,
              N_level, n_instr, wald, wald_p)
end
