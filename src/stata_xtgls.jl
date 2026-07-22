# =============================================================================
# Stata-style advanced panel: stata_xtregar (Baltagi-Wu AR(1) FE / RE), stata_xtscc
# (Driscoll-Kraay HAC SEs) and xtgls (feasible GLS long panel with panels âˆˆ
# {iid, heteroskedastic, correlated} Ã— corr âˆˆ {independent, ar1, psar1}).
# Ported from Cameron & Trivedi ch08.
# =============================================================================

"""
    stata_xtregar(df, yvar, xvars, panelvar, timevar; method=:fe)

Stata-style `stata_xtregar` — FE or RE with panel AR(1) error correction.
Estimates a common AR(1) coefficient ρ from FE within-residuals,
applies Prais-Winsten transformation, then runs FE-within or
Swamy-Arora RE on the transformed data.
"""
function stata_xtregar(df, yvar::Symbol, xvars::Vector{Symbol},
                 panelvar::Symbol, timevar::Symbol;
                 method::Symbol = :fe)   # :fe or :re
    d = DataFrames.sort(DataFrames.dropmissing(df, vcat([yvar, panelvar, timevar], xvars)),
             [panelvar, timevar])
    panels = DataFrames.sort(unique(d[!, panelvar]))
    N = length(panels)
    pidx = Dict(g => findall(==(g), d[!, panelvar]) for g in panels)

    k = length(xvars)
    y_raw = Float64.(d[!, yvar])
    X_raw = hcat(ones(DataFrames.nrow(d)), hcat([Float64.(d[!, v]) for v in xvars]...))

    # — Step 1: FE within → residuals → ρ̂ (Baltagi-Wu) —
    m_fe = FixedEffectModels.reg(d, term(yvar) ~ sum(term.(xvars)) + FixedEffectModels.fe(panelvar))
    e_fe = y_raw .- X_raw[:, 2:end] * StatsBase.coef(m_fe)
    # demean residuals within panel to get within residuals
    for g in panels
        idx = pidx[g]
        e_fe[idx] .-= Statistics.mean(e_fe[idx])
    end
    num_rho = 0.0; den_rho = 0.0
    for g in panels
        idx = pidx[g]
        ei = e_fe[idx]
        for t in 2:length(ei)
            num_rho += ei[t] * ei[t-1]
            den_rho += ei[t-1]^2
        end
    end
    ρ = clamp(num_rho / den_rho, -0.999, 0.999)

    # — Step 2: Quasi-demean (Prais-Winsten) —
    ỹ = similar(y_raw)
    X̃ = similar(X_raw)
    s1 = sqrt(1 - ρ^2)
    for g in panels
        idx = pidx[g]
        ỹ[idx[1]]      = s1 * y_raw[idx[1]]
        X̃[idx[1], :]   = s1 .* X_raw[idx[1], :]
        for t in 2:length(idx)
            ỹ[idx[t]]    = y_raw[idx[t]] - ρ * y_raw[idx[t-1]]
            X̃[idx[t], :] = X_raw[idx[t], :] .- ρ .* X_raw[idx[t-1], :]
        end
    end

    if method == :fe
        # Within transform on PW-transformed data
        for g in panels
            idx = pidx[g]
            ỹ[idx] .-= Statistics.mean(ỹ[idx])
            for j in axes(X̃, 2)
                X̃[idx, j] .-= Statistics.mean(X̃[idx, j])
            end
        end
        X_est = X̃[:, 2:end]   # drop constant (absorbed by FE)
        β = X_est \ ỹ
        u = ỹ .- X_est * β
        n_obs = length(ỹ)
        dof = n_obs - N - k
        σ2 = sum(abs2, u) / dof
        V = σ2 .* LinearAlgebra.inv(X_est' * X_est)
        se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
        # Recover constant as Statistics.mean(y - Xβ) across panel means of original data
        panel_means = DataFrames.combine(DataFrames.groupby(d, panelvar),
            yvar => (x -> Statistics.mean(Float64.(x))) => :ym,
            [v => (x -> Statistics.mean(Float64.(x))) => Symbol(v, :_m) for v in xvars]...)
        cons = Statistics.mean(panel_means.ym .- sum(β[j] * panel_means[!, Symbol(xvars[j], :_m)]
                                          for j in 1:k))
        β_full = vcat(β, [cons])
        se_full = vcat(se, [NaN])
        V_full = zeros(k+1, k+1); V_full[1:k, 1:k] = V
        cnames = vcat(string.(xvars), "_cons")
    else  # :re — Swamy-Arora on PW data
        # Estimate variance components from PW data
        # FE within on PW data
        X_pw_w = copy(X̃[:, 2:end]); ỹ_w = copy(ỹ)
        for g in panels
            idx = pidx[g]
            ỹ_w[idx] .-= Statistics.mean(ỹ_w[idx])
            for j in axes(X_pw_w, 2)
                X_pw_w[idx, j] .-= Statistics.mean(X_pw_w[idx, j])
            end
        end
        β_w = X_pw_w \ ỹ_w
        u_w = ỹ_w .- X_pw_w * β_w
        n_obs = length(ỹ)
        σ2_ε = sum(abs2, u_w) / (n_obs - N - k + 1)

        # Between on PW panel means
        Ts = [length(pidx[g]) for g in panels]
        T̄ = Statistics.mean(Ts)
        y_b = [Statistics.mean(ỹ[pidx[g]]) for g in panels]
        X_b = hcat(ones(N),
                   hcat([[Statistics.mean(X̃[pidx[g], j]) for g in panels]
                         for j in 2:size(X̃, 2)]...))
        β_b = X_b \ y_b
        u_b = y_b .- X_b * β_b
        σ2_1 = sum(abs2, u_b) / (N - k - 1)
        σ2_u = max(σ2_1 - σ2_ε / T̄, 0.0)
        θ = 1 - sqrt(σ2_ε / (T̄ * σ2_u + σ2_ε))

        # θ-demean the PW data
        ỹ_s = copy(ỹ); X̃_s = copy(X̃)
        for g in panels
            idx = pidx[g]
            ym = Statistics.mean(ỹ[idx])
            ỹ_s[idx] .-= θ * ym
            for j in axes(X̃_s, 2)
                X̃_s[idx, j] .-= θ * Statistics.mean(X̃[idx, j])
            end
        end
        β_full = X̃_s \ ỹ_s
        u_s = ỹ_s .- X̃_s * β_full
        V = σ2_ε .* LinearAlgebra.inv(X̃_s' * X̃_s)
        se_full = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
        V_full = V
        # reorder: xvars first, constant last
        order = vcat(2:k+1, 1)
        β_full = β_full[order]
        se_full = se_full[order]
        V_full = V_full[order, order]
        cnames = vcat(string.(xvars), "_cons")
    end

    return (; β = β_full, V = V_full, se = se_full,
              coefnames = cnames, n_obs = length(ỹ),
              n_panels = N, ρ = ρ, method)
end

"""
    stata_xtscc(df, yvar, xvars, panelvar, timevar; fe=false, lag=4)

Stata-style `stata_xtscc` — Driscoll-Kraay standard errors for pooled OLS (or FE
within if `fe=true`). Uses a Newey-West HAC estimator on the cross-sectional
sums of moment conditions.
"""
function stata_xtscc(df, yvar::Symbol, xvars::Vector{Symbol},
               panelvar::Symbol, timevar::Symbol;
               fe::Bool = false, lag::Int = 4)
    d = DataFrames.sort(DataFrames.dropmissing(df, vcat([yvar, panelvar, timevar], xvars)),
             [panelvar, timevar])
    y = Float64.(d[!, yvar])
    X = hcat(ones(DataFrames.nrow(d)), hcat([Float64.(d[!, v]) for v in xvars]...))

    if fe
        # Within-transform
        for g in unique(d[!, panelvar])
            idx = findall(==(g), d[!, panelvar])
            y[idx] .-= Statistics.mean(y[idx])
            for j in axes(X, 2); X[idx, j] .-= Statistics.mean(X[idx, j]); end
        end
        X_est = X[:, 2:end]   # drop constant (absorbed)
    else
        X_est = X
    end
    k = size(X_est, 2)
    β = X_est \ y
    u = y .- X_est * β
    n = length(y)

    # Driscoll-Kraay: time-based Newey-West on cross-sectional averages
    times = DataFrames.sort(unique(d[!, timevar]))
    TT = length(times)
    tidx = Dict(t => findall(==(t), d[!, timevar]) for t in times)

    # S_t = Σ_i X_it' u_it   (cross-sectional sum at each t)
    S = zeros(TT, k)
    for (ti, t) in enumerate(times)
        idx = tidx[t]
        for j in 1:k
            S[ti, j] = sum(X_est[idx, j] .* u[idx])
        end
    end

    # Newey-West on S
    Γ0 = S' * S / TT
    Ω = copy(Γ0)
    for ℓ in 1:lag
        w = 1 - ℓ / (lag + 1)  # Bartlett kernel
        Γℓ = zeros(k, k)
        for ti in (ℓ+1):TT
            Γℓ += S[ti, :] * S[ti-ℓ, :]'
        end
        Γℓ ./= TT
        Ω .+= w .* (Γℓ .+ Γℓ')
    end
    XtX_inv = LinearAlgebra.inv(X_est' * X_est)
    V = n * XtX_inv * Ω * XtX_inv   # scale by n (as Stata does)
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    if fe
        # Recover constant
        d2 = DataFrames.sort(DataFrames.dropmissing(df, vcat([yvar, panelvar, timevar], xvars)),
                  [panelvar, timevar])
        pm = DataFrames.combine(DataFrames.groupby(d2, panelvar),
            yvar => (x -> Statistics.mean(Float64.(x))) => :ym,
            [v => (x -> Statistics.mean(Float64.(x))) => Symbol(v, :_m) for v in xvars]...)
        cons = Statistics.mean(pm.ym .- sum(β[j] * pm[!, Symbol(xvars[j], :_m)] for j in 1:k))
        β_full = vcat(β, [cons])
        se_full = vcat(se, [NaN])
        V_full = zeros(k+1, k+1); V_full[1:k, 1:k] = V
        cnames = vcat(string.(xvars), "_cons")
    else
        # reorder: xvars first, constant last (constant is column 1 in X_est)
        order = vcat(2:k, 1)
        β_full = β[order]; se_full = se[order]
        V_full = V[order, order]
        cnames = vcat(string.(xvars), "_cons")
    end
    return (; β = β_full, V = V_full, se = se_full,
              coefnames = cnames, n_obs = n, fe, lag)
end

"""
    stata_xtgls(df, y, xvars; panelvar, timevar,
                panels=:correlated, corr=:psar1, level=0.95)

Stata-equivalent `xtgls` for balanced panels. Replicates

    xtset <panelvar> <timevar>
    xtgls <y> <xvars>, panels(<panels>) corr(<corr>)

Supported options:
- `panels`:  `:iid`, `:heteroskedastic`, `:correlated` (default)
- `corr`:    `:independent`, `:ar1`, `:psar1` (default)

Prints a Stata-style header + coefficient table and returns a NamedTuple with
fields `β`, `V`, `se`, `z`, `p`, `coefnames`, `n_obs`, `N_panels`, `T_periods`,
`ρ`, `Wald`, `Wald_p`, `panels`, `corr`, `y_name`.
"""
function stata_xtgls(df, y, xvars; panelvar::Symbol, timevar::Symbol,
                    panels::Symbol=:correlated, corr::Symbol=:psar1,
                    level::Float64=0.95)
    ys  = Symbol(y)
    xsv = [Symbol(v) for v in xvars]
    dfs = DataFrames.sort(df, [panelvar, timevar])
    pan = DataFrames.sort(unique(dfs[!, panelvar]))
    N   = length(pan)
    n   = DataFrames.nrow(dfs)
    T   = n ÷ N

    yv = Float64.(dfs[!, ys])
    X  = hcat(ones(n), hcat([Float64.(dfs[!, v]) for v in xsv]...))
    k  = size(X, 2)
    pidx = [((i-1)*T+1):(i*T) for i in 1:N]

    # Step 1: OLS residuals → AR(1) coefficients (Stata's rhotype=regress).
    # Note: do NOT clamp ρ — Stata allows |ρ| ≥ 1 (explosive AR1). The PW
    # first-obs weight below uses max(1-ρ², 0) which handles |ρ| ≥ 1 by
    # zeroing that panel's first observation.
    e0 = yv - X * (X \ yv)
    ρ = if corr == :independent
        zeros(N)
    elseif corr == :ar1
        num = sum(LinearAlgebra.dot(e0[pidx[i]][2:end], e0[pidx[i]][1:end-1]) for i in 1:N)
        den = sum(LinearAlgebra.dot(e0[pidx[i]][1:end-1], e0[pidx[i]][1:end-1]) for i in 1:N)
        fill(num/den, N)
    elseif corr == :psar1
        [LinearAlgebra.dot(e0[pidx[i]][2:end], e0[pidx[i]][1:end-1]) /
         LinearAlgebra.dot(e0[pidx[i]][1:end-1], e0[pidx[i]][1:end-1])
         for i in 1:N]
    else
        error("Unknown corr: $corr (use :independent, :ar1, or :psar1)")
    end

    # Step 2: Prais-Winsten transformation (if any ρ ≠ 0)
    ỹ = copy(yv); X̃ = copy(X)
    if any(!=(0.0), ρ)
        for i in 1:N
            idx = pidx[i]
            s = sqrt(max(1 - ρ[i]^2, 0.0))    # = 0 when |ρ| ≥ 1
            ỹ[idx[1]]    = s * yv[idx[1]]
            X̃[idx[1], :] = s * X[idx[1], :]
            for t in 2:T
                ỹ[idx[t]]    = yv[idx[t]] - ρ[i] * yv[idx[t-1]]
                X̃[idx[t], :] = X[idx[t], :] - ρ[i] * X[idx[t-1], :]
            end
        end
    end

    # Step 3: Cross-sectional/heteroskedastic covariance from residuals of an
    # OLS fit on the PW-transformed data.  Stata drops the first observation
    # per panel from Σ̂ when corr is ar1/psar1 (to remove the PW first-obs
    # heteroskedasticity), and uses divisor = number of obs used in Σ̂.
    e_pw      = ỹ - X̃ * (X̃ \ ỹ)
    drop_first = corr != :independent
    idx_Σ(i) = drop_first ? pidx[i][2:end] : pidx[i]
    T_eff = drop_first ? (T - 1) : T          # effective T used in Σ̂ estimation
    Σ_inv = if panels == :iid
        denom = drop_first ? (n - N - k) : (n - k)
        σ2    = sum(abs2, vcat([e_pw[idx_Σ(i)] for i in 1:N]...)) / max(denom, 1)
        Matrix{Float64}(LinearAlgebra.I(N) ./ σ2)
    elseif panels == :heteroskedastic
        σ2_i = [sum(abs2, e_pw[idx_Σ(i)]) / T_eff for i in 1:N]
        Matrix{Float64}(LinearAlgebra.Diagonal(1.0 ./ σ2_i))
    elseif panels == :correlated
        E = hcat([e_pw[idx_Σ(i)] for i in 1:N]...)    # T_eff × N
        # Stata uses divisor = T_eff (number of rows in E), not T.
        LinearAlgebra.inv(E' * E / T_eff)
    else
        error("Unknown panels: $panels (use :iid, :heteroskedastic, :correlated)")
    end

    # Step 4: GLS with Ω⁻¹ = Σ⁻¹ ⊗ I_T, applied time-period-by-time-period
    XΩX = zeros(k, k); XΩy = zeros(k)
    for t in 1:T
        rows = [(i-1)*T + t for i in 1:N]
        Xt = X̃[rows, :]; yt = ỹ[rows]
        XΩX += Xt' * Σ_inv * Xt
        XΩy += Xt' * Σ_inv * yt
    end
    β  = XΩX \ XΩy
    V  = LinearAlgebra.inv(XΩX)
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- zcrit .* se
    ci_hi = β .+ zcrit .* se

    Wald   = β[2:end]' * LinearAlgebra.inv(V[2:end, 2:end]) * β[2:end]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(k-1), Wald)

    ncov = panels == :iid             ? 1        :
           panels == :heteroskedastic ? N        :
                                        N*(N+1) ÷ 2
    nρ   = corr == :independent ? 0 :
           corr == :ar1         ? 1 :
                                  N         # psar1

    coefnames = ["(Intercept)"; string.(xsv)]

    # Stata %9.0g-style formatter (strip leading 0 for |x|<1)
    function _g(x, w; sig=7)
        (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
        s = Printf.@sprintf("%.*g", sig, x)
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return Printf.@sprintf("%*s", w, s)
    end

    panel_desc = panels == :iid             ? "homoskedastic" :
                 panels == :heteroskedastic ? "heteroskedastic" :
                                              "heteroskedastic with cross-sectional correlation"
    corr_desc  = corr == :independent ? "no autocorrelation" :
                 corr == :ar1         ? "common AR(1)"       :
                                        "panel-specific AR(1)"

    println()
    println("Cross-sectional time-series FGLS regression")
    println()
    println("Coefficients:  generalized least squares")
    println("Panels:        ", panel_desc)
    println("Correlation:   ", corr_desc)
    println()
    Printf.@printf("Estimated covariances      = %9d          Number of obs     = %10d\n",
                   ncov, n)
    Printf.@printf("Estimated autocorrelations = %9d          Number of groups  = %10d\n",
                   nρ, N)
    Printf.@printf("Estimated coefficients     = %9d          Time periods      = %10d\n",
                   k, T)
    wlabel = Printf.@sprintf("Wald chi2(%d)", k-1)
    Printf.@printf("%s%-18s= %10.2f\n", " "^48, wlabel, Wald)
    Printf.@printf("%sProb > chi2       = %10.4f\n", " "^48, Wald_p)
    println()

    println("-"^78)
    Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                   string(ys), "Coefficient", "Std. err.", "z", "P>|z|",
                   round(Int, 100*level))
    println("-"^13, "+", "-"^64)
    for j in 2:k
        Printf.@printf("%12s | %s  %s  %6.2f  %5.3f  %s  %s\n",
                       string(xsv[j-1]), _g(β[j],10), _g(se[j],9),
                       z[j], pv[j], _g(ci_lo[j],11), _g(ci_hi[j],11))
    end
    Printf.@printf("%12s | %s  %s  %6.2f  %5.3f  %s  %s\n",
                   "_cons", _g(β[1],10), _g(se[1],9),
                   z[1], pv[1], _g(ci_lo[1],11), _g(ci_hi[1],11))
    println("-"^78)

    return (β=β, V=V, se=se, z=z, p=pv, coefnames=coefnames,
            n_obs=n, N_panels=N, T_periods=T, ρ=ρ,
            Wald=Wald, Wald_p=Wald_p, panels=panels, corr=corr, y_name=ys)
end
