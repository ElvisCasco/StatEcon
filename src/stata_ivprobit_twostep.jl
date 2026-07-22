# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_ivprobit_twostep — Newey (1987) two-step / control-function IV probit
# --------------------------------------------------------------------------

"""
    stata_ivprobit_twostep(df; depvar, exog_vars, endog, instruments,
                           level=0.95, quiet=false) -> NamedTuple

Stata-style `ivprobit <depvar> <exog_vars> (<endog> = <instruments>),
twostep first` — Newey (1987) two-step / control-function IV probit.

Pipeline:
  1. **First stage**: OLS of `<endog>` on `[exog_vars; instruments]`
     (with intercept), printed via `stata_regress`.
  2. Compute first-stage residual  v̂_i = y1_i − z_iᵀπ̂.
  3. **Second stage**: probit of `<depvar>` on `[exog_vars; <endog>; v̂]` via
     `stata_probit`. The probit coefficients on `[exog_vars; <endog>]` are the
     two-step IV-probit estimates of the structural equation.
  4. Wald test of exogeneity: `(z-stat on v̂)² ~ χ²(1)`.

The reported second-stage SEs are the augmented-probit OIM SEs (i.e. treating
v̂ as a regressor) rather than Newey's exact two-step correction. Point
estimates and the Wald exogeneity statistic match Stata's `twostep` exactly;
the SEs are an approximation that ignores first-stage variability.

Returns `(; first_stage, β_struct, se_struct, V_struct, γ, V_aug, λ̂, se_λ,
Wald_exog, Wald_exog_p, Wald, Wald_p, n, model_second)`.
"""
function stata_ivprobit_twostep(df; depvar::Symbol,
                                exog_vars::AbstractVector{Symbol},
                                endog::Symbol,
                                instruments::AbstractVector{Symbol},
                                level::Float64 = 0.95,
                                quiet::Bool = false)
    needed = unique(vcat(depvar, endog, exog_vars, instruments))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    keep = trues(DataFrames.nrow(dfc))
    for c in needed
        col = dfc[!, c]
        eltype(col) <: Real || continue
        for i in eachindex(col)
            keep[i] &= isfinite(col[i])
        end
    end
    dfc = dfc[keep, :]

    # ── First stage: OLS y1 on (exog + IV) ────────────────────────────
    fs_rhs   = vcat(exog_vars, instruments)
    fs_form  = StatsModels.term(endog) ~ sum(StatsModels.term.(fs_rhs))
    m_first  = FixedEffectModels.reg(dfc, fs_form)
    quiet || stata_regress(m_first; yname = string(endog))
    yhat     = FixedEffectModels.predict(m_first, dfc)
    dfc.vhat_ivp = Float64.(dfc[!, endog]) .- yhat

    # ── Second stage: probit y2 on (exog + endog + vhat) ─────────────
    ss_rhs   = vcat(exog_vars, endog, :vhat_ivp)
    ss_form  = StatsModels.term(depvar) ~ sum(StatsModels.term.(ss_rhs))
    m_aug    = stata_probit(dfc, ss_form; quiet = true)
    cn_aug   = m_aug.coefnames_glm                          # GLM order
    β_aug    = m_aug.β_glm
    V_aug    = m_aug.V_glm
    n_obs    = m_aug.n

    # Indices for structural equation (everything except v̂):
    idx_v    = findfirst(==("vhat_ivp"), cn_aug)
    idx_int  = findfirst(==("(Intercept)"), cn_aug)
    idx_keep = setdiff(1:length(β_aug), [idx_v])      # struct: x..., endog, _cons
    # Stata-display order: exog_vars in formula order, then endog, then _cons.
    order_struct = Int[]
    for v in exog_vars
        push!(order_struct, findfirst(==(string(v)), cn_aug))
    end
    push!(order_struct, findfirst(==(string(endog)), cn_aug))
    push!(order_struct, idx_int)

    β_struct  = β_aug[order_struct]
    V_struct  = V_aug[order_struct, order_struct]
    se_struct = sqrt.(max.(LinearAlgebra.diag(V_struct), 0.0))

    # Wald test of exogeneity = (z on vhat)²
    se_λ = sqrt(max(V_aug[idx_v, idx_v], 0.0))
    λ̂   = β_aug[idx_v]
    Wald_exog   = (λ̂ / se_λ)^2
    Wald_exog_p = 1 - Distributions.cdf(Distributions.Chisq(1), Wald_exog)

    # Wald chi² on slopes for header (excludes _cons, includes endog).
    slope_idx = 1:length(order_struct) - 1
    Wald_struct = β_struct[slope_idx]' *
                  LinearAlgebra.inv(V_struct[slope_idx, slope_idx]) *
                  β_struct[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                   Wald_struct)

    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    z_struct  = β_struct ./ se_struct
    p_struct  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_struct)))
    ci_lo     = β_struct .- crit .* se_struct
    ci_hi     = β_struct .+ crit .* se_struct

    # ── Print second-stage block ──────────────────────────────────────
    function g9(x; w::Int=10, sig::Int=7)
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
        nstr = commafmt(n_obs)
        wstr = Printf.@sprintf("%.2f", Wald_struct)
        pstr = Printf.@sprintf("%.4f", Wald_p)
        vw = maximum(length, (nstr, wstr, pstr))
        Printf.@printf("%-50s%-15s = %*s\n",
                       "Two-step probit with endogenous regressors",
                       "Number of obs", vw, nstr)
        Printf.@printf("%50s%-15s = %*s\n", "",
                       "Wald chi2($(length(slope_idx)))", vw, wstr)
        Printf.@printf("%50s%-15s = %*s\n", "", "Prob > chi2", vw, pstr)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100*level)
        println("-"^13, "+", "-"^64)
        labels = vcat([string(v) for v in exog_vars], string(endog), "_cons")
        for i in eachindex(order_struct)
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           labels[i], g9(β_struct[i]; w=10), g9(se_struct[i]; w=9),
                           Printf.@sprintf("%7.2f", z_struct[i]),
                           Printf.@sprintf("%.3f", p_struct[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^78)
        Printf.@printf("Wald test of exogeneity: chi2(1) = %.2f%19sProb > chi2 = %.4f\n",
                       Wald_exog, "", Wald_exog_p)
        println("Endogenous: ", string(endog))
        exo_full = vcat([string(v) for v in exog_vars],
                        [string(v) for v in instruments])
        prefix = "Exogenous:  "
        line   = prefix
        for w in exo_full
            if length(line) + length(w) + 1 > 78
                println(line)
                line = "            " * w
            else
                line *= (line == prefix ? "" : " ") * w
            end
        end
        println(line)
    end

    return (; first_stage = m_first,
              β_struct, se_struct, V_struct,
              γ = β_struct[length(exog_vars) + 1],
              V_aug, λ̂, se_λ,
              Wald_exog, Wald_exog_p,
              Wald = Wald_struct, Wald_p,
              n = n_obs, model_second = m_aug.model)
end
