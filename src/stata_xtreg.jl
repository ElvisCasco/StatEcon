# =============================================================================
# Stata-style panel regressions: xtreg (pa, re) + stata_xtreg_re Stata-format printer,
# stata_hausman, areg. Plus private helpers _panel_r2s and _coef_of / _coefnames_of /
# _vcov_of / _nobs_of (used by stata_hausman + estimates_table).
# Ported from Cameron & Trivedi ch08.
# =============================================================================

# Private helpers matching StatsBase / NamedTuple duck-typing used by stata_hausman.
_coef_of(m)      = hasproperty(m, :β) ? m.β      : StatsBase.coef(m)
_coefnames_of(m) = hasproperty(m, :coefnames) ? string.(m.coefnames) :
                                                string.(StatsBase.coefnames(m))
_vcov_of(m)      = hasproperty(m, :V) ? m.V      : StatsBase.vcov(m)
_nobs_of(m)      = hasproperty(m, :n_obs) ? m.n_obs :
                                            try StatsBase.nobs(m) catch; missing end

"""
    _panel_r2s(df, y, xs, idvar, β_slopes)

Stata-style overall, between, and within panel R²s: `cor(y, Xβ̂)²` at the
observation, panel-mean, and within-demeaned levels respectively. Returns
`(; r2_o, r2_b, r2_w)`. `β_slopes` must match `xs` in order (no intercept).
"""
function _panel_r2s(df, y::Symbol, xs::AbstractVector{Symbol}, idvar::Symbol,
                   β_slopes::AbstractVector{<:Real})
    d  = DataFrames.dropmissing(df, unique(vcat(y, xs, idvar)))
    y_vec = Float64.(_sm_rawval.(d[!, y]))
    X  = hcat([Float64.(_sm_rawval.(d[!, v])) for v in xs]...)
    gm = DataFrames.combine(DataFrames.groupby(d, idvar),
        (v => (x -> Statistics.mean(Float64.(_sm_rawval.(x)))) =>
              Symbol(v, "_m")
         for v in vcat(y, xs))...)
    dj = DataFrames.leftjoin(d, gm, on=idvar)
    y_bar = Float64.(dj[!, Symbol(y, "_m")])
    X_bar = hcat([Float64.(dj[!, Symbol(v, "_m")]) for v in xs]...)
    y_b   = Float64.(gm[!, Symbol(y, "_m")])
    X_b   = hcat([Float64.(gm[!, Symbol(v, "_m")]) for v in xs]...)
    _safe(a, b) = Statistics.std(a) < 1e-12 || Statistics.std(b) < 1e-12 ?
                   NaN : Statistics.cor(a, b)^2
    return (; r2_o = _safe(y_vec, X * β_slopes),
              r2_b = _safe(y_b,   X_b * β_slopes),
              r2_w = _safe(y_vec .- y_bar, (X .- X_bar) * β_slopes))
end

"""
    stata_xtreg_pa(df, y, xvars, idvar, timevar; corr=:ar, lags=2,
             max_iter=20, tol=1e-8, vce=:robust)

Stata-style `xtreg ..., pa corr(...) vce(robust)` — population-averaged (GEE)
estimator of Liang & Zeger (1986), described in Cameron & Trivedi (2005) §8.4.

Working correlation structures supported:
  * `:independent`  — identity R (→ pooled OLS)
  * `:exchangeable` — R_ts = ρ for t ≠ s
  * `:ar`           — AR(`lags`) errors; autocovariances built via Yule-Walker
                      so R is fully populated

The algorithm iterates:
  1. Compute Pearson residuals from the current β̂.
  2. Estimate ρ̂ (or φ̂ for AR(p)) from residuals.
  3. Build the T × T working correlation R(ρ̂) and V_i = σ̂² R.
  4. FGLS update β̂ = (Σᵢ X_i' V_i⁻¹ X_i)⁻¹ Σᵢ X_i' V_i⁻¹ y_i
until ‖Δβ‖ < tol or `max_iter` iterations.

`vce=:robust` returns the Liang-Zeger cluster-robust sandwich:
  V_β = A⁻¹ B A⁻¹,   A = Σᵢ X_i' V_i⁻¹ X_i,   B = Σᵢ X_i' V_i⁻¹ u_i u_i' V_i⁻¹ X_i.

Works with balanced or unbalanced panels.
"""
function stata_xtreg_pa(df, y::Symbol, xvars::AbstractVector{Symbol},
                  idvar::Symbol, timevar::Symbol;
                  corr::Symbol=:ar, lags::Int=2,
                  max_iter::Int=20, tol::Float64=1e-8,
                  vce::Symbol=:robust)
    d = DataFrames.dropmissing(
        DataFrames.sort(df, [idvar, timevar]),
        vcat([y, idvar, timevar], xvars))

    panels = DataFrames.groupby(d, idvar)
    pd = [(y = Vector{Float64}(g[!, y]),
           X = hcat(ones(DataFrames.nrow(g)),
                    Matrix{Float64}(g[:, xvars])))
          for g in panels]
    N_panels = length(pd)
    N_obs    = sum(length(p.y) for p in pd)
    k        = size(pd[1].X, 2)
    Tmax     = maximum(length(p.y) for p in pd)
    cnames   = vcat(["_cons"], string.(xvars))

    # ---- Initial OLS on stacked data -----------------------------------------
    X_all = reduce(vcat, [p.X for p in pd])
    y_all = reduce(vcat, [p.y for p in pd])
    β     = X_all \ y_all

    # ---- Helpers -------------------------------------------------------------
    build_R(ρvec::Vector, T::Int) =
        [i == j ? 1.0 : ρvec[abs(i - j)] for i in 1:T, j in 1:T]

    function estimate_rho(residuals_by_panel, σ2)
        if corr == :independent
            return zeros(Tmax)
        elseif corr == :exchangeable
            num, den = 0.0, 0
            for u in residuals_by_panel
                T_i = length(u)
                for i in 1:T_i, j in (i+1):T_i
                    num += u[i] * u[j]
                    den += 1
                end
            end
            ρ = (num / den) / σ2
            return fill(ρ, Tmax - 1)         # index 1..Tmax-1 for lags 1..Tmax-1
        elseif corr == :ar
            ρ̂ = zeros(lags)
            for ℓ in 1:lags
                num, den = 0.0, 0
                for u in residuals_by_panel
                    T_i = length(u)
                    for t in 1:T_i-ℓ
                        num += u[t] * u[t+ℓ]
                        den += 1
                    end
                end
                ρ̂[ℓ] = (num / den) / σ2
            end
            # Yule-Walker: solve R_matrix · φ = ρ̂  (R_matrix[i,j] = ρ̂_|i-j|)
            Rm = [i == j ? 1.0 :
                  abs(i-j) <= lags ? ρ̂[abs(i-j)] : 0.0
                  for i in 1:lags, j in 1:lags]
            # Need to fill in the inner lags from ρ̂ only (no lower lags of ρ̂)
            for i in 1:lags, j in 1:lags
                d_ = abs(i - j)
                Rm[i, j] = d_ == 0 ? 1.0 : ρ̂[d_]
            end
            φ = Rm \ ρ̂
            # Extend autocorrelation to all lags via AR recursion
            ρ_full = zeros(Tmax - 1)
            ρ_full[1:lags] = ρ̂
            for ℓ in (lags+1):(Tmax - 1)
                ρ_full[ℓ] = sum(φ[i] * ρ_full[ℓ - i] for i in 1:lags if ℓ - i >= 1;
                                init = 0.0)
            end
            return ρ_full
        else
            error("Unknown corr structure: $corr")
        end
    end

    # ---- Iteration -----------------------------------------------------------
    ρ_full = zeros(Tmax - 1)
    σ2     = 1.0
    for iter in 1:max_iter
        u_by_panel = [p.y .- p.X * β for p in pd]
        σ2 = sum(sum(abs2, u) for u in u_by_panel) / (N_obs - k)
        ρ_full = estimate_rho(u_by_panel, σ2)

        A = zeros(k, k); rhs = zeros(k)
        for p in pd
            T_i = length(p.y)
            Ri  = build_R(ρ_full, T_i)
            Vi_inv = LinearAlgebra.inv(σ2 * Ri)
            A   .+= p.X' * Vi_inv * p.X
            rhs .+= p.X' * Vi_inv * p.y
        end
        β_new = A \ rhs
        LinearAlgebra.norm(β_new - β) < tol && (β = β_new; break)
        β = β_new
    end

    # ---- Sandwich (Liang-Zeger robust) SE ------------------------------------
    A = zeros(k, k); B = zeros(k, k)
    for p in pd
        T_i = length(p.y)
        Ri  = build_R(ρ_full, T_i)
        Vi_inv = LinearAlgebra.inv(σ2 * Ri)
        A   .+= p.X' * Vi_inv * p.X
        u    = p.y .- p.X * β
        B   .+= p.X' * Vi_inv * (u * u') * Vi_inv * p.X
    end
    V = vce == :robust ? (LinearAlgebra.inv(A) * B * LinearAlgebra.inv(A)) :
                         σ2 .* LinearAlgebra.inv(A)
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))

    # ---- Report --------------------------------------------------------------
    corrlabel = corr == :ar            ? "AR($lags)" :
                corr == :exchangeable  ? "exchangeable" :
                                          "independent"
    println("GEE population-averaged model  (xtreg, pa corr($corrlabel) vce($vce))")
    Printf.@printf("  N obs    = %d\n  N panels = %d\n  T (max)  = %d\n  σ̂²     = %.6f\n",
                   N_obs, N_panels, Tmax, σ2)
    Printf.@printf("  correlation = %s\n", corrlabel)
    println(DataFrames.DataFrame(variable = cnames, estimate = β,
                                 stderr = se, z = z, p = pv))
    # Report R matrix (Stata's `matrix list e(R)`)
    R_eR = build_R(ρ_full, Tmax)
    println("\ne(R) — estimated working correlation matrix:")
    stata_matlist(R_eR, "R"; symmetric = false)
    return (; β, V, se, coefnames = cnames, σ2, ρ = ρ_full, R = R_eR,
              n_obs = N_obs, n_panels = N_panels)
end

"""
    stata_xtreg_re(df, y, xvars, idvar; vce=:classical)

Stata-style `xtreg y x1 x2 ..., re` — GLS random-effects estimator with the
**Swamy-Arora** variance-component estimator (Stata's default). This is a
from-scratch implementation, so results match Stata's `xtreg, re` (not the
`xtreg, re mle` option or MixedModels' REML).

Algorithm (C&T 2005 §9.3; Baltagi 2013 §2.3):

  1. Fit FE within regression → σ̂²_ε = RSS_FE / (NT − N − k + 1)
  2. Fit BE (between) on panel means → σ̂²₁ = RSS_BE / (N − k)
  3. σ̂²_u = max(σ̂²₁ − σ̂²_ε / T̄, 0)
  4. θ̂ = 1 − √(σ̂²_ε / (T̄ σ̂²_u + σ̂²_ε))
  5. Form y* = y − θ̂ȳᵢ, X* = X − θ̂X̄ᵢ, and OLS on the θ-demeaned data

`vce` options:  `:classical` (default, GLS),  `:cluster` (cluster-robust on id).
Returns a NamedTuple; pass it to `stata_xtreg_re_print` for a Stata-style print.
"""
function stata_xtreg_re(df, y::Symbol, xvars::AbstractVector{Symbol}, idvar::Symbol;
                  vce::Symbol=:classical)
    d = DataFrames.dropmissing(df, vcat([y, idvar], xvars))
    n_obs = DataFrames.nrow(d)
    cnames = vcat(string.(xvars), "_cons")

    # ---- Step 1: FE within → σ²_ε ----
    fe_formula = term(y) ~ sum(term.(xvars)) + FixedEffectModels.fe(idvar)
    m_fe = FixedEffectModels.reg(d, fe_formula)
    n_panels = length(unique(d[!, idvar]))
    # Compute FE residuals manually (avoid the missing-value issue in residuals(m, df))
    # y_it - ȳ_i and x_it - x̄_i — then FE β applied to within y
    means = DataFrames.combine(DataFrames.groupby(d, idvar),
        (v => (x -> Statistics.mean(Float64.(_sm_rawval.(collect(skipmissing(x)))))) => Symbol(v, "_bar")
         for v in vcat([y], xvars))...)
    d2 = DataFrames.leftjoin(d, means, on=idvar)
    for v in vcat([y], xvars)
        d2[!, Symbol(v, "_w")] = Float64.(d2[!, v]) .- d2[!, Symbol(v, "_bar")]
    end
    y_w = Vector{Float64}(d2[!, Symbol(y, "_w")])
    X_w = Matrix{Float64}(d2[:, Symbol.(string.(xvars) .* "_w")])
    β_fe = X_w \ y_w
    u_fe = y_w - X_w * β_fe
    k    = length(xvars)
    σ2_ε = sum(abs2, u_fe) / (n_obs - n_panels - k)

    # ---- Step 2: BE on panel means → σ²₁ ----
    be_df = DataFrames.combine(DataFrames.groupby(d, idvar),
        (v => (x -> Statistics.mean(Float64.(_sm_rawval.(collect(skipmissing(x)))))) => v
         for v in vcat([y], xvars))...)
    y_b = Vector{Float64}(be_df[!, y])
    X_b = hcat(ones(n_panels), Matrix{Float64}(be_df[:, xvars]))
    β_be = X_b \ y_b
    u_be = y_b - X_b * β_be
    σ2_1 = sum(abs2, u_be) / (n_panels - k - 1)           # per-obs variance at panel-mean level

    T_obs = DataFrames.combine(DataFrames.groupby(d, idvar), DataFrames.nrow => :T)
    T̄    = Statistics.mean(T_obs.T)
    σ2_u  = max(σ2_1 - σ2_ε / T̄, 0.0)
    θ     = 1 - sqrt(σ2_ε / (T̄ * σ2_u + σ2_ε))

    # ---- Step 3: θ-demean and OLS ----
    for v in vcat([y], xvars)
        d2[!, Symbol(v, "_s")] = Float64.(d2[!, v]) .- θ .* d2[!, Symbol(v, "_bar")]
    end
    y_s = Vector{Float64}(d2[!, Symbol(y, "_s")])
    X_s = hcat(Matrix{Float64}(d2[:, Symbol.(string.(xvars) .* "_s")]),
               fill(1 - θ, n_obs))                       # θ-transformed constant = (1-θ)
    β    = X_s \ y_s
    u    = y_s .- X_s * β

    # ---- Step 4: Variance-covariance ----
    XtX_inv = LinearAlgebra.inv(X_s' * X_s)
    if vce == :classical
        σ2_u_hat = σ2_ε          # under RE, Var(u*) = σ²_ε (θ-demeaning normalizes the variance)
        V = σ2_u_hat .* XtX_inv
    elseif vce == :cluster
        # Liang-Zeger on the θ-demeaned data
        B = zeros(size(X_s, 2), size(X_s, 2))
        for g in unique(d[!, idvar])
            idx = findall(==(g), d[!, idvar])
            Xi  = X_s[idx, :]
            ui  = u[idx]
            B  .+= Xi' * (ui * ui') * Xi
        end
        V = XtX_inv * B * XtX_inv
        # Small-sample adj a la Stata: G/(G-1) * (N-1)/(N-k)
        G = n_panels; N = n_obs
        V .*= (G / (G - 1)) * ((N - 1) / (N - size(X_s, 2)))
    else
        error("Unknown vce: $vce   (use :classical or :cluster)")
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))

    # ── Stata-style header statistics
    β_slopes  = β[1:end-1]
    V_slopes  = V[1:end-1, 1:end-1]
    Wald      = β_slopes' * LinearAlgebra.inv(V_slopes) * β_slopes
    Wald_p    = 1 - Distributions.cdf(Distributions.Chisq(length(β_slopes)), Wald)
    r2s       = _panel_r2s(d, y, xvars, idvar, β_slopes)
    T_counts  = T_obs.T
    T_min, T_max = minimum(T_counts), maximum(T_counts)

    return (; β, se, z, p=pv, V, coefnames=cnames,
              σ_ε = sqrt(σ2_ε), σ_u = sqrt(σ2_u),
              ρ   = σ2_u / (σ2_u + σ2_ε),
              θ, T̄, T_min, T_max,
              r2_within  = r2s.r2_w,
              r2_between = r2s.r2_b,
              r2_overall = r2s.r2_o,
              Wald, Wald_p,
              n_obs, n_panels, vce, y_name = y)
end

"""
    stata_xtreg_re_print(res; level=0.95)

Pretty-print a NamedTuple produced by `stata_xtreg_re` in Stata's exact layout.
"""
function stata_xtreg_re_print(res; level::Float64=0.95)
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    lo   = res.β .- crit .* res.se
    hi   = res.β .+ crit .* res.se
    q    = length(res.β) - 1                          # # slopes

    _fmtn(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")
    _g(x, w) = begin
        (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
        s = Printf.@sprintf("%.7g", x)
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        Printf.@sprintf("%*s", w, s)
    end

    # ── Header block
    Printf.@printf("%-48s%-18s= %10s\n", "Random-effects GLS regression",
                   "Number of obs", _fmtn(res.n_obs))
    Printf.@printf("%-48s%-18s= %10s\n", "Group variable: id",
                   "Number of groups", _fmtn(res.n_panels))
    println()
    Printf.@printf("%-48s%s\n", "R-squared:", "Obs per group:")
    Printf.@printf("%-48s%-18s= %10d\n",
                   Printf.@sprintf("     Within  = %.4f", res.r2_within),
                   "                  min", res.T_min)
    Printf.@printf("%-48s%-18s= %10.1f\n",
                   Printf.@sprintf("     Between = %.4f", res.r2_between),
                   "                  avg", res.T̄)
    Printf.@printf("%-48s%-18s= %10d\n",
                   Printf.@sprintf("     Overall = %.4f", res.r2_overall),
                   "                  max", res.T_max)
    println()
    Printf.@printf("%-48s%-18s= %10.2f\n", "",
                   "Wald chi2($q)", res.Wald)
    Printf.@printf("%-48s%-18s= %10.4f\n",
                   "corr(u_i, X) = 0 (assumed)", "Prob > chi2", res.Wald_p)
    println()

    # ── Coefficient table (z, P>|z|)
    lvl  = round(Int, 100 * level)
    println("-"^78)
    Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                   string(res.y_name), "Coefficient", "Std. err.",
                   "z", "P>|z|", lvl)
    println("-"^13, "+", "-"^64)
    for i in eachindex(res.β)
        nm = res.coefnames[i]
        Printf.@printf("%12s | %s  %s  %6.2f  %5.3f  %s  %s\n",
                       nm, _g(res.β[i], 10), _g(res.se[i], 9),
                       res.z[i], res.p[i],
                       _g(lo[i], 11), _g(hi[i], 10))
    end
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s\n", "sigma_u", _g(res.σ_u, 10))
    Printf.@printf("%12s | %s\n", "sigma_e", _g(res.σ_ε, 10))
    Printf.@printf("%12s | %s   (fraction of variance due to u_i)\n",
                   "rho", _g(res.ρ, 10))
    println("-"^78)
    return nothing
end

"""
    stata_hausman(m_fe, m_re; sigmamore=false)

Stata-style `stata_hausman FE RE[, sigmamore]`. Tests
   H₀: RE is consistent and efficient (so `cov(β̂_FE − β̂_RE) = V_FE − V_RE`)
   H₁: FE is consistent, RE is not.

Accepts either `StatsBase`-compatible models (e.g. `FixedEffectModels.reg`)
or the NamedTuple outputs of `stata_xtreg_re` / `hausman_taylor` / `arellano_bond`.
The intercept / `_cons` column is excluded from the comparison (matching
Stata's convention).

If `V_FE − V_RE` is not positive-definite, falls back to the Moore-Penrose
pseudo-inverse. With `sigmamore=true`, uses the FE residual variance to
rescale `V_RE` so that the difference is more likely to be PD
(analogous to Stata's `sigmamore` option).

Prints the per-coefficient (b) / (B) / (b-B) / sqrt(diag(V_b−V_B)) table
that Stata outputs, and returns a NamedTuple with the stats.
"""
function stata_hausman(m_fe, m_re; sigmamore::Bool=false)
    nm_fe = _coefnames_of(m_fe)
    nm_re = _coefnames_of(m_re)
    # Intersect, drop intercept
    common = [v for v in intersect(nm_fe, nm_re)
              if !(v in ("_cons", "(Intercept)"))]
    i_fe = [findfirst(==(v), nm_fe) for v in common]
    i_re = [findfirst(==(v), nm_re) for v in common]
    β_fe = _coef_of(m_fe)[i_fe]
    β_re = _coef_of(m_re)[i_re]
    V_fe = _vcov_of(m_fe)[i_fe, i_fe]
    V_re = _vcov_of(m_re)[i_re, i_re]

    # Stata's sigmamore: rebuild V_FE using σ²_RE (common "efficient" variance)
    # so that V_FE − V_RE has the form σ²_RE[(X'QX)⁻¹ − (X*'X*)⁻¹], which is PSD.
    # σ²_RE comes from `stata_xtreg_re.σ_ε²`; σ²_FE is read off the FixedEffectModels
    # fit via RSS/dof_residual — and since those two `dof_residual` conventions
    # aren't always identical, we take the ratio so V_FE is *explicitly* scaled
    # to use σ²_RE regardless of the underlying fit.
    function _σ2(m)
        hasproperty(m, :σ_ε)          && return m.σ_ε^2
        hasproperty(m, :rss) && hasproperty(m, :dof_residual) &&
            return m.rss / m.dof_residual
        try
            return StatsBase.deviance(m) / StatsBase.dof_residual(m)
        catch
            return missing
        end
    end
    if sigmamore
        σ2_fe, σ2_re = _σ2(m_fe), _σ2(m_re)
        if !ismissing(σ2_fe) && !ismissing(σ2_re) && σ2_fe > 0
            V_fe = (σ2_re / σ2_fe) .* V_fe      # rescale V_FE to use σ²_RE
        end
    end

    ΔV = V_fe - V_re
    # Drop rows with non-finite β or non-finite DIAGONAL of ΔV. (Checking the
    # full row would reject every coefficient if any column has a NaN
    # — which happens when the FE model absorbs a collinear regressor
    # such as a time-invariant variable.)
    bad = findall(k -> !isfinite(β_fe[k]) || !isfinite(β_re[k]) ||
                       !isfinite(ΔV[k, k]),
                  eachindex(common))
    if !isempty(bad)
        @warn "stata_hausman: dropping non-finite rows for $(common[bad])"
        keep   = setdiff(eachindex(common), bad)
        β_fe   = β_fe[keep];  β_re = β_re[keep]
        ΔV     = ΔV[keep, keep]
        common = common[keep]
    end
    # Scrub any residual NaN/Inf off-diagonals from dropping rows/cols.
    ΔV = replace(ΔV, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    d = β_fe .- β_re
    H = try
        d' * (ΔV \ d)
    catch
        @warn "V_FE − V_RE not positive-definite; using pseudoinverse."
        d' * LinearAlgebra.pinv(ΔV) * d
    end
    q = length(d)
    pv = 1 - Distributions.cdf(Distributions.Chisq(max(q, 1)), H)

    # Stata-style layout (pipe-separated, %.7f-ish with stripped leading 0s)
    se_diff = sqrt.(max.(LinearAlgebra.diag(ΔV), 0.0))

    _g(x, w) = begin
        (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
        s = Printf.@sprintf("%.7f", x)
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        Printf.@sprintf("%*s", w, s)
    end

    println("                 ---- Coefficients ----")
    println("             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))")
    println("             |       FE           RE         Difference       Std. err.")
    println("-"^13, "+", "-"^64)
    for k in eachindex(common)
        Printf.@printf("%12s | %10s   %10s       %10s       %10s\n",
                       common[k], _g(β_fe[k], 10), _g(β_re[k], 10),
                       _g(d[k], 10),  _g(se_diff[k], 10))
    end
    println("-"^78)
    println("                          b = Consistent under H0 and Ha; obtained from xtreg.")
    println("           B = Inconsistent under Ha, efficient under H0; obtained from xtreg.")
    println()
    println("Test of H0: Difference in coefficients not systematic")
    println()
    Printf.@printf("    chi2(%d) = (b-B)'[(V_b-V_B)^(-1)](b-B)\n", q)
    Printf.@printf("            = %.2f\n", H)
    Printf.@printf("Prob > chi2 = %6.4f\n", pv)
    return (; chi2=H, df=q, p=pv, diff=d, se_diff, vars=common)
end

"""
    stata_areg(df, y, xs; absorb, cluster=nothing, level=0.95)

Stata-style `areg y xs…, absorb(id) [vce(cluster id)]` — LSDV / fixed-
effects regression with the panel indicators absorbed. Fit via
`FixedEffectModels.reg(... + fe(absorb), Vcov.cluster(cluster))` (or
`Vcov.simple()` when `cluster=nothing`). `xs` variables that are
collinear with the absorbed fixed effect (e.g. time-invariant regressors)
are flagged `(omitted)` in the coefficient table.

Prints the Stata `areg` header block — `Linear regression, absorbing
indicators`, `Absorbed variable`, F, Prob > F, R², Adj R², Root MSE — the
cluster banner, and the coefficient table. `_cons = ȳ − x̄'β̂`, with SE by
the delta method on the cluster vcov.
"""
function stata_areg(df, y, xs::AbstractVector; absorb::Symbol,
                    cluster::Union{Symbol,Nothing}=nothing,
                    level::Float64=0.95)
    ys  = Symbol(y)
    xsv = [Symbol(v) for v in xs]
    needed = unique(vcat(ys, xsv, absorb))
    cluster !== nothing && push!(needed, cluster)
    d = DataFrames.dropmissing(df, unique(needed))
    n = DataFrames.nrow(d)

    f    = term(ys) ~ sum(term.(xsv)) + FixedEffectModels.fe(absorb)
    vcov = cluster === nothing ? FixedEffectModels.Vcov.simple() :
                                 FixedEffectModels.Vcov.cluster(cluster)
    m    = FixedEffectModels.reg(d, f, vcov)

    β_all    = StatsBase.coef(m)
    V_all    = StatsBase.vcov(m)
    cn_all   = string.(StatsBase.coefnames(m))

    # FixedEffectModels keeps dropped (collinear) regressors in the output
    # with NaN coef and NaN in the corresponding row AND column of V. Filter
    # on the DIAGONAL only — the off-diagonals linking a bad column to a good
    # row are NaN too, so row-wise `all(isfinite, …)` would reject every row.
    ok   = findall(i -> isfinite(β_all[i]) && isfinite(V_all[i, i]),
                   eachindex(β_all))
    β    = β_all[ok]
    V    = V_all[ok, ok]
    # Safety: any remaining non-finite cross terms collapse to 0 (those rows
    # are orthogonal to everything else the sandwich can say anything about).
    V    = replace(V, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    cn_m = cn_all[ok]

    n_groups = length(unique(d[!, absorb]))
    rss   = StatsBase.deviance(m)
    r2    = StatsBase.r2(m)
    ar2   = StatsBase.adjr2(m)
    rmse  = sqrt(rss / Int(StatsBase.dof_residual(m)))

    # For cluster inference, df = #clusters − 1.
    if cluster !== nothing
        cluster_N = length(unique(d[!, cluster]))
        dof_r     = cluster_N - 1
    else
        cluster_N = 0
        dof_r     = Int(StatsBase.dof_residual(m))
    end

    q     = length(β)
    Fstat = q > 0 ? (β' * LinearAlgebra.inv(V) * β) / q : NaN
    pF    = q > 0 ? 1 - Distributions.cdf(Distributions.FDist(q, dof_r), Fstat) :
                     NaN

    tcrit  = Distributions.quantile(Distributions.TDist(dof_r), 1 - (1-level)/2)
    t_st   = β ./ se
    pvals  = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(dof_r), abs.(t_st)))
    ci_lo  = β .- tcrit .* se
    ci_hi  = β .+ tcrit .* se

    # _cons = ȳ − x̄'β̂ (only valid β), SE via delta method on the cluster vcov.
    y_mean  = Statistics.mean(Float64.(_sm_rawval.(d[!, ys])))
    x_means = Float64[Statistics.mean(Float64.(_sm_rawval.(d[!, Symbol(nm)])))
                      for nm in cn_m]
    cons_b  = y_mean - LinearAlgebra.dot(x_means, β)
    cons_se = sqrt(max(x_means' * V * x_means, 0.0))
    cons_t  = cons_se > 0 ? cons_b / cons_se : NaN
    cons_p  = 2 * (1 - Distributions.cdf(Distributions.TDist(dof_r), abs(cons_t)))
    cons_lo = cons_b - tcrit * cons_se
    cons_hi = cons_b + tcrit * cons_se

    dropped = [v for v in string.(xsv) if !(v in cn_m)]

    _fmtn(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")
    _g(x, w; sig=7) = begin
        (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
        s = Printf.@sprintf("%.*g", sig, x)
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        Printf.@sprintf("%*s", w, s)
    end

    # ── Header
    Printf.@printf("%-52s%-18s= %6s\n",
                   "Linear regression, absorbing indicators",
                   "Number of obs", _fmtn(n))
    Printf.@printf("%-52s%-18s= %6s\n",
                   "Absorbed variable: $absorb",
                   "No. of categories", _fmtn(n_groups))
    Printf.@printf("%-52s%-18s= %6.2f\n", "", "F($q, $dof_r)",  Fstat)
    Printf.@printf("%-52s%-18s= %6.4f\n", "", "Prob > F",       pF)
    Printf.@printf("%-52s%-18s= %6.4f\n", "", "R-squared",      r2)
    Printf.@printf("%-52s%-18s= %6.4f\n", "", "Adj R-squared",  ar2)
    Printf.@printf("%-52s%-18s= %6.4f\n", "", "Root MSE",       rmse)
    println()
    if cluster !== nothing
        Printf.@printf("%78s\n",
            "(Std. err. adjusted for $cluster_N clusters in $cluster)")
    end

    # ── Coefficient table
    lvl = round(Int, 100 * level)
    println("-"^78)
    Printf.@printf("%12s | %22s\n", "", "Robust")
    Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                   string(ys), "Coefficient", "std. err.",
                   "t", "P>|t|", lvl)
    println("-"^13, "+", "-"^64)
    for v in string.(xsv)
        if v in dropped
            Printf.@printf("%12s | %10d  (omitted)\n", v, 0)
        else
            idx = findfirst(==(v), cn_m)
            Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                           v, _g(β[idx], 10), _g(se[idx], 9),
                           t_st[idx], pvals[idx],
                           _g(ci_lo[idx], 11), _g(ci_hi[idx], 10))
        end
    end
    Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                   "_cons", _g(cons_b, 10), _g(cons_se, 9),
                   cons_t, cons_p, _g(cons_lo, 11), _g(cons_hi, 10))
    println("-"^78)

    return (; β, V, se, coefnames = cn_m, dropped,
              r2, adj_r2 = ar2, rmse,
              F = Fstat, df1 = q, df2 = dof_r, pF,
              n_obs = n, n_groups, cluster_N,
              _cons = cons_b, _cons_se = cons_se)
end
