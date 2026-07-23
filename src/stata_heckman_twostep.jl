# (deps provided by the StatEcon module)

"""
    stata_heckman_twostep(df, outcome, selection; depvar_y="lny",
                          depvar_d="dy", level=0.95, quiet=false) -> NamedTuple

Stata `heckman <outcome>, select(<selection>) twostep` — the Heckman (1979)
two-step estimator with the first-stage-corrected variance:

  Stage 1: probit(`selection`) on the full sample → γ̂,
           inverse Mills ratio λ_i = φ(z_i'γ̂)/Φ(z_i'γ̂)
  Stage 2: OLS(`outcome` + λ) on the selected subsample → α̂ = (β̂, β̂_λ),
           σ̂² = ε̂'ε̂/n₂ + β̂_λ²·mean(δ), δ_i = λ_i(λ_i + z_i'γ̂),  ρ̂ = β̂_λ/σ̂.

The outcome-block VCE uses Greene's stage-1 correction
`V(α̂) = σ̂²(X*'X*)⁻¹{X*'(I − ρ̂²Δ)X* + ρ̂²X*'ΔZ V_γ Z'ΔX*}(X*'X*)⁻¹`,
with V_γ the probit Fisher-information inverse. Prints Stata's three-block
output (outcome / selection / /mills) plus ρ̂ and σ̂.

`outcome` and `selection` are `StatsModels` formulas; `depvar_y` / `depvar_d`
name the outcome and 0/1 selection columns (defaults `"lny"` / `"dy"`).

Returns `(; β, se, V, coefnames, rho, sigma, lambda, n, …)` with `β`/`se`/`V`
the outcome block (constant last), `lambda` the inverse-Mills coefficient,
plus the selection block `β_selection`/`se_selection`.
"""
function stata_heckman_twostep(df, outcome::StatsModels.FormulaTerm,
                               selection::StatsModels.FormulaTerm;
                               depvar_y::AbstractString = "lny",
                               depvar_d::AbstractString = "dy",
                               level::Float64 = 0.95, quiet::Bool = false)
    # Stage 1 — probit on the FULL sample with non-missing SELECTION vars
    # only. The outcome variable is legitimately missing for unselected
    # rows, so it must NOT enter the dropmissing here.
    sel_vars = StatsModels.termvars(selection)
    out_vars = StatsModels.termvars(outcome)
    needed   = unique(vcat(sel_vars, out_vars))
    df_h = DataFrames.dropmissing(df[:, needed], sel_vars)
    n_full = DataFrames.nrow(df_h)

    m_pr = GLM.glm(selection, df_h,
                   Distributions.Binomial(), GLM.ProbitLink())
    γ̂   = GLM.coef(m_pr); cn_pr = GLM.coefnames(m_pr)
    Z   = GLM.modelmatrix(m_pr)
    xb_full = Z * γ̂
    λ_full  = Distributions.pdf.(Distributions.Normal(), xb_full) ./
              Distributions.cdf.(Distributions.Normal(), xb_full)
    δ_full  = λ_full .* (λ_full .+ xb_full)

    d_full = Int.(df_h[!, Symbol(depvar_d)])
    is_pos = Bool[d == 1 for d in d_full]
    n_pos  = sum(is_pos); n_zero = n_full - n_pos

    # Stage 2 — OLS on selected obs, with λ as an additional regressor.
    df_pos = df_h[is_pos, :]
    df_pos.invmills = λ_full[is_pos]
    out_xs = out_vars[2:end]                    # skip lhs
    X_pos  = hcat(ones(n_pos),
                 [Float64.(df_pos[!, v]) for v in out_xs]...,
                 df_pos.invmills)
    y_pos  = Float64.(df_pos[!, Symbol(depvar_y)])
    α̂      = X_pos \ y_pos
    ε̂      = y_pos .- X_pos * α̂

    β̂_λ    = α̂[end]
    δ_pos  = δ_full[is_pos]
    σ̂²     = (ε̂'ε̂) / n_pos + β̂_λ^2 * Statistics.mean(δ_pos)
    σ̂      = sqrt(σ̂²)
    ρ̂      = β̂_λ / σ̂

    safe_sym_inv(A; rtol::Float64 = 1e-10) = begin
        As = LinearAlgebra.Symmetric(A)
        try
            LinearAlgebra.inv(As)
        catch
            LinearAlgebra.pinv(Matrix(As); rtol = rtol)
        end
    end

    XX_inv = safe_sym_inv(X_pos' * X_pos)
    Φ_full = clamp.(Distributions.cdf.(Distributions.Normal(), xb_full),
                    1e-12, 1 - 1e-12)
    φ_full = Distributions.pdf.(Distributions.Normal(), xb_full)
    w_full = (φ_full .^ 2) ./ (Φ_full .* (1 .- Φ_full))
    A_γ    = Z' * LinearAlgebra.Diagonal(w_full) * Z          # probit Fisher info
    Vγ     = safe_sym_inv(A_γ)
    Z_pos  = Z[is_pos, :]
    D_pos  = LinearAlgebra.Diagonal(δ_pos)
    M1     = X_pos' * (LinearAlgebra.I(n_pos) - ρ̂^2 .* D_pos) * X_pos
    M2     = ρ̂^2 .* (X_pos' * D_pos * Z_pos) * Vγ * (Z_pos' * D_pos * X_pos)
    V_α    = σ̂² .* XX_inv * (M1 + M2) * XX_inv
    se_α   = sqrt.(max.(LinearAlgebra.diag(V_α), 0.0))

    # α̂ order: (Intercept), out_xs..., invmills. Reorder for Stata display.
    k_out  = length(out_xs)
    α̂_disp = vcat(α̂[2:k_out+1], [α̂[1]])
    se_disp = vcat(se_α[2:k_out+1], [se_α[1]])
    α_lam  = α̂[end]; se_lam = se_α[end]

    z_disp = α̂_disp ./ se_disp
    p_disp = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_disp)))
    crit   = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    lo_disp = α̂_disp .- crit .* se_disp
    hi_disp = α̂_disp .+ crit .* se_disp

    z_lam = α_lam / se_lam
    p_lam = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_lam)))
    lo_lam = α_lam - crit * se_lam; hi_lam = α_lam + crit * se_lam

    # Observed information, matching Stata's vce(oim) and `stata_probit`.
    # `GLM.vcov` here is the expected (IRLS) information, which left the
    # selection-block standard errors ~2% above Stata's.
    se_pr = sqrt.(max.(LinearAlgebra.diag(
        _probit_oim_vcov(Z, Float64.(GLM.response(m_pr)), γ̂)), 0.0))
    z_pr  = γ̂ ./ se_pr
    p_pr  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_pr)))
    lo_pr = γ̂ .- crit .* se_pr; hi_pr = γ̂ .+ crit .* se_pr
    i_int_pr = findfirst(==("(Intercept)"), cn_pr)
    slope_pr = setdiff(1:length(γ̂), [i_int_pr])
    ord_pr   = vcat(slope_pr, [i_int_pr])
    cn_pr_ord = vcat(cn_pr[slope_pr], "_cons")
    γ̂_disp = γ̂[ord_pr]; se_pr_disp = se_pr[ord_pr]
    z_pr_disp = z_pr[ord_pr]; p_pr_disp = p_pr[ord_pr]
    lo_pr_disp = lo_pr[ord_pr]; hi_pr_disp = hi_pr[ord_pr]

    βs = α̂_disp[1:end-1]
    Vs = V_α[2:k_out+1, 2:k_out+1]
    Wald = try
        (βs' * (LinearAlgebra.Symmetric(Vs) \ βs))::Float64
    catch
        (βs' * LinearAlgebra.pinv(Vs) * βs)::Float64
    end
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(k_out), Wald)

    function g9(x; w::Int = 10, sig::Int = 7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w)
    end
    commafmt(num) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    if !quiet
        println()
        Printf.@printf("%-48s%-13s = %10s\n",
                       "Heckman selection model -- two-step estimates",
                       "Number of obs", commafmt(n_full))
        Printf.@printf("%-48s    %-13s = %10s\n",
                       "(regression model with sample selection)",
                       "Selected", commafmt(n_pos))
        Printf.@printf("%-48s    %-13s = %10s\n",
                       "", "Nonselected", commafmt(n_zero))
        println()
        Printf.@printf("%48s%-13s = %10.2f\n", "",
                       "Wald chi2($(k_out))", Wald)
        Printf.@printf("%48s%-13s = %10.4f\n", "", "Prob > chi2", Wald_p)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100*level)
        println("-"^13, "+", "-"^64)

        Printf.@printf("%-12s |\n", depvar_y)
        for i in 1:k_out
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           string(out_xs[i]), g9(α̂_disp[i]; w=10), g9(se_disp[i]; w=9),
                           Printf.@sprintf("%7.2f", z_disp[i]),
                           Printf.@sprintf("%.3f", p_disp[i]),
                           g9(lo_disp[i]; w=9), g9(hi_disp[i]; w=10))
        end
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "_cons", g9(α̂_disp[end]; w=10), g9(se_disp[end]; w=9),
                       Printf.@sprintf("%7.2f", z_disp[end]),
                       Printf.@sprintf("%.3f", p_disp[end]),
                       g9(lo_disp[end]; w=9), g9(hi_disp[end]; w=10))
        println("-"^13, "+", "-"^64)

        Printf.@printf("%-12s |\n", depvar_d)
        for i in eachindex(γ̂_disp)
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           cn_pr_ord[i], g9(γ̂_disp[i]; w=10), g9(se_pr_disp[i]; w=9),
                           Printf.@sprintf("%7.2f", z_pr_disp[i]),
                           Printf.@sprintf("%.3f", p_pr_disp[i]),
                           g9(lo_pr_disp[i]; w=9), g9(hi_pr_disp[i]; w=10))
        end
        println("-"^13, "+", "-"^64)

        Printf.@printf("%-12s |\n", "/mills")
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "lambda", g9(α_lam; w=10), g9(se_lam; w=9),
                       Printf.@sprintf("%7.2f", z_lam),
                       Printf.@sprintf("%.3f", p_lam),
                       g9(lo_lam; w=9), g9(hi_lam; w=10))
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %10s\n", "rho",   Printf.@sprintf("%9.5f",  ρ̂))
        Printf.@printf("%12s | %10s\n", "sigma", Printf.@sprintf("%9.7f", σ̂))
        println("-"^78)
    end

    return (; β = α̂_disp, se = se_disp, V = V_α,
              coefnames = vcat(string.(out_xs), ["_cons"]),
              rho = ρ̂, sigma = σ̂, lambda = α_lam, n = n_full,
              β_outcome = α̂_disp, se_outcome = se_disp, V_outcome = V_α,
              β_selection = γ̂_disp, se_selection = se_pr_disp,
              β_lambda = α_lam, se_lambda = se_lam, σ = σ̂, ρ = ρ̂,
              n_pos = n_pos, n_zero = n_zero,
              outcome_coefnames = vcat(string.(out_xs), ["_cons"]),
              selection_coefnames = cn_pr_ord)
end
