# ============================================================================
# stata_arellano_bond.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_arellano_bond(df, y, idvar, timevar; lags=1, xvars, twostep, maxldep, pre, endog)

Arellano-Bond (1991) difference GMM for a dynamic linear panel model, with
one-step or two-step (Windmeijer-corrected) robust standard errors.
"""
function stata_arellano_bond(df, y::Symbol, idvar::Symbol, timevar::Symbol;
                       lags::Int=1, xvars::Vector{Symbol}=Symbol[],
                       twostep::Bool=false, maxldep::Int=0,
                       pre::Vector{Tuple{Symbol,Int,Int}}=Tuple{Symbol,Int,Int}[],
                       endog::Vector{Tuple{Symbol,Int,Int}}=Tuple{Symbol,Int,Int}[])
    all_syms = unique(vcat([y, idvar, timevar], xvars,
                           [v for (v,_,_) in pre], [v for (v,_,_) in endog]))
    d = DataFrames.sort(DataFrames.dropmissing(df, all_syms), [idvar, timevar])
    grouped = DataFrames.groupby(d, idvar)
    N_groups = length(grouped)

    n_pre_regs   = sum(a + 1 for (_,a,_) in pre;   init=0)
    n_endog_regs = sum(a + 1 for (_,a,_) in endog;  init=0)
    k_reg = lags + n_pre_regs + n_endog_regs + length(xvars)

    gmm_groups = Tuple{Symbol, Int, Int}[]
    push!(gmm_groups, (y, lags, maxldep > 0 ? maxldep : 999))
    for (v, a, b) in pre;    push!(gmm_groups, (v, a + 1, b));  end
    for (v, a, b) in endog;  push!(gmm_groups, (v, a + 2, b));  end

    t_start = lags + 2
    T_max = maximum(DataFrames.nrow(g) for g in grouped)
    n_periods = max(0, T_max - t_start + 1)
    period_info = Vector{Vector{Tuple{Int,Int}}}(undef, n_periods)
    col = 1
    for (pidx, t) in enumerate(t_start:T_max)
        info = Tuple{Int,Int}[]
        for (_, s, depth) in gmm_groups
            n_inst = max(0, min(depth, t - s))
            push!(info, (col, n_inst))
            col += n_inst
        end
        period_info[pidx] = info
    end
    gmm_ncol = col - 1
    n_z = gmm_ncol + length(xvars)

    panel_dy = Vector{Vector{Float64}}()
    panel_dx = Vector{Matrix{Float64}}()
    panel_zi = Vector{Matrix{Float64}}()
    T_vals   = Int[]

    for g in grouped
        T_i = DataFrames.nrow(g)
        n_use = max(0, T_i - t_start + 1)
        n_use < 1 && continue
        y_v = collect(Float64, map(_c9_rawval, g[!, y]))
        pre_vecs   = [collect(Float64, map(_c9_rawval, g[!, v])) for (v,_,_) in pre]
        endog_vecs = [collect(Float64, map(_c9_rawval, g[!, v])) for (v,_,_) in endog]
        xvar_vecs  = [collect(Float64, map(_c9_rawval, g[!, v])) for v in xvars]
        all_gmm_vecs = vcat([y_v], pre_vecs, endog_vecs)

        dy_i = zeros(n_use)
        dx_i = zeros(n_use, k_reg)
        zi_i = zeros(n_use, n_z)

        for (row, t) in enumerate(t_start:T_i)
            dy_i[row] = y_v[t] - y_v[t-1]

            c = 0
            for k in 1:lags
                c += 1
                dx_i[row, c] = (t-k-1 >= 1) ? y_v[t-k] - y_v[t-k-1] : 0.0
            end
            for (pi, (_, a, _)) in enumerate(pre)
                for k in 0:a
                    c += 1
                    dx_i[row, c] = (t-k-1 >= 1) ? pre_vecs[pi][t-k] - pre_vecs[pi][t-k-1] : 0.0
                end
            end
            for (ei, (_, a, _)) in enumerate(endog)
                for k in 0:a
                    c += 1
                    dx_i[row, c] = (t-k-1 >= 1) ? endog_vecs[ei][t-k] - endog_vecs[ei][t-k-1] : 0.0
                end
            end
            for xi in eachindex(xvars)
                c += 1
                dx_i[row, c] = (t-1 >= 1) ? xvar_vecs[xi][t] - xvar_vecs[xi][t-1] : 0.0
            end

            pidx = t - t_start + 1
            for (gi, (_, s, _)) in enumerate(gmm_groups)
                cs, n_inst = period_info[pidx][gi]
                for j in 1:n_inst
                    idx_src = t - s - n_inst + j
                    zi_i[row, cs + j - 1] = (idx_src >= 1) ? all_gmm_vecs[gi][idx_src] : 0.0
                end
            end

            for xi in eachindex(xvars)
                zi_i[row, gmm_ncol + xi] = (t-1 >= 1) ? xvar_vecs[xi][t] - xvar_vecs[xi][t-1] : 0.0
            end
        end
        push!(panel_dy, dy_i);  push!(panel_dx, dx_i)
        push!(panel_zi, zi_i);  push!(T_vals, n_use)
    end
    isempty(panel_dy) && error("No usable observations for Arellano-Bond.")

    yvec = vcat(panel_dy...)
    Xmat = vcat(panel_dx...)
    Zmat_raw = vcat(panel_zi...)

    nz_cols = findall(c -> any(!iszero, Zmat_raw[:, c]), 1:size(Zmat_raw, 2))
    Zmat = Zmat_raw[:, nz_cols]
    panel_zi = [zi[:, nz_cols] for zi in panel_zi]
    n_z = size(Zmat, 2)

    N_obs = length(yvec)
    N_ind = length(panel_dy)

    ZHZ = zeros(n_z, n_z)
    for i in 1:N_ind
        Ti = T_vals[i]
        Hi = zeros(Ti, Ti)
        for r in 1:Ti
            Hi[r, r] = 2.0
            r < Ti && (Hi[r, r+1] = -1.0)
            r > 1  && (Hi[r, r-1] = -1.0)
        end
        ZHZ .+= panel_zi[i]' * Hi * panel_zi[i]
    end
    W1 = LinearAlgebra.pinv(ZHZ)
    XZ = Xmat' * Zmat
    β1 = (XZ * W1 * XZ') \ (XZ * W1 * Zmat' * yvec)

    β = β1
    if twostep
        u1 = yvec .- Xmat * β1
        Ω1 = zeros(n_z, n_z);  offset = 0
        for i in 1:N_ind
            Ti = T_vals[i];  ui = u1[offset+1:offset+Ti]
            Ω1 .+= panel_zi[i]' * (ui * ui') * panel_zi[i]
            offset += Ti
        end
        W2 = LinearAlgebra.pinv(Ω1)
        β = (XZ * W2 * XZ') \ (XZ * W2 * Zmat' * yvec)

        u2 = yvec .- Xmat * β
        Ω2 = zeros(n_z, n_z);  off2 = 0
        for i in 1:N_ind
            Ti = T_vals[i];  ui = u2[off2+1:off2+Ti]
            Ω2 .+= panel_zi[i]' * (ui * ui') * panel_zi[i]
            off2 += Ti
        end
        A2 = XZ * W2 * XZ'
        A2inv = LinearAlgebra.inv(A2)
        V2_robust = A2inv * (XZ * W2 * Ω2 * W2 * XZ') * A2inv

        A1 = XZ * W1 * XZ'
        A1inv = LinearAlgebra.inv(A1)
        V1_robust = A1inv * (XZ * W1 * Ω1 * W1 * XZ') * A1inv

        Zpu2 = Zmat' * u2
        a_vec = W2 * Zpu2
        c_vec = Zmat * a_vec
        M     = A2inv * (XZ * W2) * Zmat'
        weight = u1 .* c_vec
        D     = -2.0 .* M * LinearAlgebra.Diagonal(weight) * Xmat

        Vβ = V2_robust + D * V2_robust + V2_robust * D' + D * V1_robust * D'
    else
        u = yvec .- Xmat * β
        Ω = zeros(n_z, n_z);  offset = 0
        for i in 1:N_ind
            Ti = T_vals[i];  ui = u[offset+1:offset+Ti]
            Ω .+= panel_zi[i]' * (ui * ui') * panel_zi[i]
            offset += Ti
        end
        A = XZ * W1 * XZ'
        Ainv = LinearAlgebra.inv(A)
        Vβ = Ainv * (XZ * W1 * Ω * W1 * XZ') * Ainv
    end

    y_sum = 0.0;  yl_sum = zeros(k_reg);  n_lev = 0
    for g in grouped
        T_i = DataFrames.nrow(g);  T_i < lags + 1 && continue
        y_v = collect(Float64, map(_c9_rawval, g[!, y]))
        pre_vecs   = [collect(Float64, map(_c9_rawval, g[!, v])) for (v,_,_) in pre]
        endog_vecs = [collect(Float64, map(_c9_rawval, g[!, v])) for (v,_,_) in endog]
        xvar_vecs  = [collect(Float64, map(_c9_rawval, g[!, v])) for v in xvars]
        for t in (lags+1):T_i
            y_sum += y_v[t]
            c = 0
            for k in 1:lags;  c += 1;  yl_sum[c] += y_v[t-k];  end
            for (pi, (_, a, _)) in enumerate(pre)
                for k in 0:a;  c += 1;  yl_sum[c] += pre_vecs[pi][t-k];  end
            end
            for (ei, (_, a, _)) in enumerate(endog)
                for k in 0:a;  c += 1;  yl_sum[c] += endog_vecs[ei][t-k];  end
            end
            for xi in eachindex(xvars);  c += 1;  yl_sum[c] += xvar_vecs[xi][t];  end
            n_lev += 1
        end
    end
    ȳ = y_sum / n_lev
    x̄_lev = yl_sum ./ n_lev
    β_cons = ȳ - LinearAlgebra.dot(β, x̄_lev)
    se_cons = sqrt(x̄_lev' * Vβ * x̄_lev)

    β_full = vcat(β, β_cons)
    V_full = zeros(k_reg + 1, k_reg + 1)
    V_full[1:k_reg, 1:k_reg] .= Vβ
    g_cons = -x̄_lev
    V_full[end, end] = se_cons^2
    cov_β_cons = vec(Vβ * g_cons)
    V_full[1:k_reg, end] .= cov_β_cons
    V_full[end, 1:k_reg] .= cov_β_cons

    se = sqrt.(max.(LinearAlgebra.diag(V_full), 0.0))
    z  = β_full ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))

    wald = β' * LinearAlgebra.inv(Vβ) * β
    wald_p = 1 - Distributions.cdf(Distributions.Chisq(k_reg), wald)

    n_instruments = n_z + 1

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

    step_lbl = twostep ? "Two-step" : "One-step"
    T_avg = N_obs / N_ind
    T_min = minimum(T_vals);  T_max_v = maximum(T_vals)

    ci_lo = β_full .- 1.96 .* se
    ci_hi = β_full .+ 1.96 .* se

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

    println()
    Printf.@printf("%-48s%-18s= %10s\n",
                   "Arellano-Bond dynamic panel-data estimation",
                   "Number of obs", commafmt(N_obs))
    Printf.@printf("Group variable: %-32s%-18s= %10d\n",
                   string(idvar), "Number of groups", N_ind)
    Printf.@printf("Time variable: %s\n", string(timevar))
    println("                                                Obs per group:")
    Printf.@printf("%48s%-5s = %10d\n", "", "min", T_min)
    Printf.@printf("%48s%-5s = %10d\n", "", "avg", round(Int, T_avg))
    Printf.@printf("%48s%-5s = %10d\n", "", "max", T_max_v)
    println()
    Printf.@printf("Number of instruments = %6d%18sWald chi2(%d)      = %10.2f\n",
                   n_instruments, "", k_reg, wald)
    Printf.@printf("%48sProb > chi2       = %10.4f\n", "", wald_p)
    println("$(step_lbl) results")
    Printf.@printf("%79s\n", "(Std. err. adjusted for clustering on $(string(idvar)))")
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
                       g9(β_full[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", pv[i]),
                       g9(ci_lo[i]; w=10), g9(ci_hi[i]; w=10))
    end

    idx = 1
    if lags > 0
        Printf.@printf("%12s |\n", string(y))
        for k in 1:lags
            prow("L$(k).", idx);  idx += 1
        end
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
        Printf.@printf("%12s |\n", string(v))
        for k in 0:a
            lbl = k == 0 ? "--." : "L$(k)."
            prow(lbl, idx);  idx += 1
        end
        println("             |")
    end
    for v in xvars
        prow(string(v), idx);  idx += 1
    end
    prow("_cons", length(β_full))
    println("-"^78)

    println("Instruments for differenced equation")
    gmm_labels = String[]
    maxlag_y = maxldep > 0 ? string(lags + maxldep - 1) : "."
    push!(gmm_labels, "L($(lags)/$(maxlag_y)).$(y)")
    for (v, a, b) in pre
        lag_pref = a > 0 ? "L$(a > 1 ? string(a) : "")." : ""
        push!(gmm_labels, "L(1/$(b)).$(lag_pref)$(v)")
    end
    for (v, a, b) in endog
        lag_pref = a > 0 ? "L$(a > 1 ? string(a) : "")." : ""
        push!(gmm_labels, "L(2/$(1+b)).$(lag_pref)$(v)")
    end
    println("        GMM-type: ", join(gmm_labels, " "))
    if !isempty(xvars)
        println("        Standard: ", join(["D." * string(v) for v in xvars], " "))
    end
    println("Instruments for level equation")
    println("        Standard: _cons")

    W_est = twostep ? W2 : W1
    return (; β=β_full, V=V_full, coefnames=colnames,
              residuals=yvec .- Xmat*β,
              Zmat, Xmat, y=yvec, twostep, W=W_est,
              panel_zi, panel_dy, T_vals, N_obs, N_ind)
end
