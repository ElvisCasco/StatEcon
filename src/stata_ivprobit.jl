# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_ivprobit — `ivprobit y x (endog = iv)` joint-MLE IV probit
# --------------------------------------------------------------------------
import Optim
import ForwardDiff

"""
    stata_ivprobit(df; depvar, exog_vars, endog, instruments,
                   vce=:oim, level=0.95, quiet=false) -> NamedTuple

Stata-style `ivprobit <depvar> <exog_vars> (<endog> = <instruments>)
[, vce(robust)] nolog` — Probit model with one continuous endogenous
regressor, joint MLE.

Model:
    y2_i = 1{x_iᵀβ + γ·y1_i + u_i > 0},          (probit / structural)
    y1_i = z_iᵀπ + δᵀIV_i + v_i,                  (linear / first-stage)
    (u_i, v_i) ∼ N(0, [[1, ρσ_v]; [ρσ_v, σ_v²]])

Estimation: MLE in (β, γ, π, δ, lnσ_v, atanh(ρ)). LBFGS with an explicit
ForwardDiff gradient, warm-started from (probit β; OLS first-stage π,δ; σ_v
from OLS residuals; ρ=0). OIM vcov from a finite-difference Hessian. For
`vce=:robust` we use the sandwich  V = A⁻¹ B A⁻¹ ·n/(n-1)  with A = -H/n and
B = (1/n)Σ s_i s_iᵀ from per-obs scores.

Output mirrors Stata: header, structural-equation coefficient table (slopes
+ endog + _cons), auxiliary block (`corr(e.<endog>, e.<depvar>)` and
`sd(e.<endog>)` with delta-method CIs), the Wald test of exogeneity
(corr = 0), and the trailing `Endogenous:` / `Exogenous:` lines.

Returns `(; β, se, V, γ, π, σ_v, ρ, ln_σv, at_ρ, se_σv, se_ρ, σv_lo, σv_hi,
ρ_lo, ρ_hi, ll, n, Wald, Wald_p, Wald_exog, Wald_exog_p, coefnames_struct,
coefnames_first, model, V_full)`.
"""
function stata_ivprobit(df; depvar::Symbol,
                        exog_vars::AbstractVector{Symbol},
                        endog::Symbol,
                        instruments::AbstractVector{Symbol},
                        vce::Symbol = :oim, level::Float64 = 0.95,
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

    y2 = Float64.(dfc[!, depvar])
    y1 = Float64.(dfc[!, endog])
    Xm = hcat(ones(DataFrames.nrow(dfc)),
              [Float64.(dfc[!, v]) for v in exog_vars]...)         # (1, x...)
    Zm = hcat(ones(DataFrames.nrow(dfc)),
              [Float64.(dfc[!, v]) for v in exog_vars]...,
              [Float64.(dfc[!, v]) for v in instruments]...)        # (1, x..., iv...)
    n   = size(Xm, 1)
    k_x = size(Xm, 2)                # _cons + exog_vars
    k_z = size(Zm, 2)                # _cons + exog_vars + instruments
    k_iv = length(instruments)
    cn_struct = vcat([string(v) for v in exog_vars], string(endog), "_cons")
    cn_first  = vcat("_cons", [string(v) for v in exog_vars],
                     [string(v) for v in instruments])

    # ── Initial values ─────────────────────────────────────────────
    # First-stage OLS: y1 ~ Z
    π0_full = Zm \ y1
    v0      = y1 .- Zm * π0_full
    σv0     = max(Statistics.std(v0; corrected = false), 1e-3)
    # Probit warm-start for structural equation: y2 ~ (X, y1)
    Wm = hcat(Xm, y1)                                              # (1, x..., y1)
    m_pr = try
        GLM.glm(Wm, y2, Distributions.Binomial(), GLM.ProbitLink())
    catch
        # Fall back to logit warm-start rescaled if probit IRLS diverges
        m_lo = GLM.glm(Wm, y2, Distributions.Binomial(), GLM.LogitLink())
        GLM.glm(Wm, y2, Distributions.Binomial(), GLM.ProbitLink();
                start = GLM.coef(m_lo) ./ 1.81)
    end
    β_struct0 = GLM.coef(m_pr)                                     # (β_x..., γ)

    # Pack: θ = [β_struct (k_x+1); π_full (k_z); ln σ_v; atanh ρ]
    p_β = k_x + 1
    p_π = k_z
    p_total = p_β + p_π + 2

    function negll(θ)
        β_s   = θ[1:p_β]                            # length k_x + 1: x..., y1
        π_fs  = θ[p_β + 1 : p_β + p_π]              # length k_z (first-stage π)
        ln_σ  = θ[p_β + p_π + 1]
        at_ρ  = θ[p_β + p_π + 2]
        σ_v   = exp(ln_σ)
        ρ_loc = tanh(at_ρ)
        η_y2  = Wm * β_s                            # x_iᵀβ + γ·y1_i
        v     = y1 .- Zm * π_fs
        denom = sqrt(1 - ρ_loc^2 + eps())
        s     = (2 .* y2 .- 1) .* (η_y2 .+ (ρ_loc / σ_v) .* v) ./ denom
        Φs    = Distributions.cdf.(Distributions.Normal(), s)
        # log(φ(v/σ_v)) - log(σ_v): log normal density of v.
        # NOTE: use `log(2 * Base.pi)` — `π` is shadowed by `π_fs` in scope.
        ll_v = -0.5 .* (v ./ σ_v) .^ 2 .- 0.5 * log(2 * Base.pi) .- ln_σ
        # log of probit term, clamped to avoid -Inf when Φs is tiny.
        ll_y = log.(max.(Φs, 1e-300))
        return -sum(ll_v .+ ll_y)
    end

    θ0 = vcat(β_struct0, π0_full, log(σv0), 0.0)
    # Optim's `autodiff = :forward` kwarg pipeline is broken in the installed
    # Optim; wrap `negll` with an explicit ForwardDiff gradient and use the
    # 5-arg optimize form.
    g!(G, x) = ForwardDiff.gradient!(G, negll, x)
    res = Optim.optimize(negll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-7, iterations = 2000))
    θ̂  = Optim.minimizer(res)
    ll = -negll(θ̂)

    β_s_hat = θ̂[1:p_β]
    π̂       = θ̂[p_β + 1 : p_β + p_π]
    ln_σv   = θ̂[p_β + p_π + 1]
    at_ρ    = θ̂[p_β + p_π + 2]
    σ_v     = exp(ln_σv)
    ρ       = tanh(at_ρ)

    # ── Hessian / vcov ─────────────────────────────────────────────
    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x)
            xpi[i] += h_[i]; xmi[i] -= h_[i]
            fpi, fmi = f(xpi), f(xmi)
            H[i, i] = (fpi - 2*f0 + fmi) / h_[i]^2
            for j in (i+1):nθ
                xpp = copy(x); xpp[i] += h_[i]; xpp[j] += h_[j]
                xpm = copy(x); xpm[i] += h_[i]; xpm[j] -= h_[j]
                xmp = copy(x); xmp[i] -= h_[i]; xmp[j] += h_[j]
                xmm = copy(x); xmm[i] -= h_[i]; xmm[j] -= h_[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h_[i] * h_[j])
            end
        end
        return H
    end
    H = _fd_hessian(negll, θ̂)
    A = LinearAlgebra.Symmetric(H)
    V_oim = LinearAlgebra.inv(A)
    V_full = if vce == :robust
        # Per-obs scores via central finite differences.
        function negll_i(θ, i)
            β_s = θ[1:p_β]; π_fs = θ[p_β+1:p_β+p_π]
            ln_σ = θ[p_β+p_π+1]; at_ρ_loc = θ[p_β+p_π+2]
            σ_v_loc = exp(ln_σ); ρ_loc = tanh(at_ρ_loc)
            η_y2 = LinearAlgebra.dot(Wm[i, :], β_s)
            v_i  = y1[i] - LinearAlgebra.dot(Zm[i, :], π_fs)
            denom = sqrt(1 - ρ_loc^2 + eps())
            s_i  = (2*y2[i] - 1) * (η_y2 + (ρ_loc / σ_v_loc) * v_i) / denom
            Φs   = Distributions.cdf(Distributions.Normal(), s_i)
            ll_v = -0.5 * (v_i / σ_v_loc)^2 - 0.5 * log(2 * Base.pi) - ln_σ
            ll_y = log(max(Φs, 1e-300))
            return -(ll_v + ll_y)
        end
        h_g = sqrt(sqrt(eps(Float64))) .* max.(abs.(θ̂), 1.0)
        # Sum of outer products of per-obs scores.
        B = zeros(p_total, p_total)
        sc = zeros(p_total)
        for i in 1:n
            for j in 1:p_total
                xp = copy(θ̂); xp[j] += h_g[j]
                xm = copy(θ̂); xm[j] -= h_g[j]
                sc[j] = (negll_i(xp, i) - negll_i(xm, i)) / (2 * h_g[j])
            end
            B .+= sc * sc'
        end
        # Sandwich: V_oim · B · V_oim, multiplied by n/(n-1) correction
        V_oim * B * V_oim * (n / (n - 1))
    else
        V_oim
    end

    se_full = sqrt.(max.(LinearAlgebra.diag(V_full), 0.0))
    se_β    = se_full[1:p_β]
    se_lnσ  = se_full[p_β + p_π + 1]
    se_atρ  = se_full[p_β + p_π + 2]

    # Reorder structural β to Stata-display order:
    # exog_vars... in formula order, then endog (linc), then _cons.
    # Currently β_s_hat = (_cons, exog_vars..., endog).
    n_x = length(exog_vars)
    ord_b = vcat(2:n_x+1, [n_x + 2], [1])
    β_disp  = β_s_hat[ord_b]
    se_disp = se_β[ord_b]
    V_disp  = V_full[1:p_β, 1:p_β][ord_b, ord_b]

    # Wald chi² on slopes (excludes _cons, INCLUDES endog).
    slope_idx_disp = 1:length(ord_b)-1   # everything except last (_cons)
    Wald = β_disp[slope_idx_disp]' *
           LinearAlgebra.inv(V_disp[slope_idx_disp, slope_idx_disp]) *
           β_disp[slope_idx_disp]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx_disp)), Wald)

    # Wald test of exogeneity (corr = 0): test atanh(ρ) = 0.
    Wald_exog   = (at_ρ / se_atρ)^2
    Wald_exog_p = 1 - Distributions.cdf(Distributions.Chisq(1), Wald_exog)

    # Delta-method SEs and CIs for σ_v and ρ (transform CI ends back).
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    z_disp = β_disp ./ se_disp
    p_disp = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_disp)))
    ci_lo  = β_disp .- crit .* se_disp
    ci_hi  = β_disp .+ crit .* se_disp

    σv_lo = exp(ln_σv - crit * se_lnσ)
    σv_hi = exp(ln_σv + crit * se_lnσ)
    ρ_lo  = tanh(at_ρ - crit * se_atρ)
    ρ_hi  = tanh(at_ρ + crit * se_atρ)
    se_σv = σ_v * se_lnσ                   # delta-method on exp
    se_ρ  = (1 - ρ^2) * se_atρ             # delta-method on tanh

    # ── Print ──────────────────────────────────────────────────────
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
        nstr = commafmt(n)
        wstr = Printf.@sprintf("%.2f", Wald)
        pstr = Printf.@sprintf("%.4f", Wald_p)
        vw = maximum(length, (nstr, wstr, pstr))
        Printf.@printf("%-56s%-13s = %*s\n",
                       "Probit model with endogenous regressors",
                       "Number of obs", vw, nstr)
        Printf.@printf("%56s%-13s = %*s\n", "",
                       "Wald chi2($(length(slope_idx_disp)))", vw, wstr)
        ll_label = vce == :robust ? "Log pseudolikelihood" : "Log likelihood"
        ll_str = Printf.@sprintf("%s = %.4f", ll_label, ll)
        right  = Printf.@sprintf("%-13s = %*s", "Prob > chi2", vw, pstr)
        pad_h  = max(0, 78 - length(ll_str) - length(right))
        println(ll_str, " "^pad_h, right)
        println()

        println("-"^78)
        if vce == :robust
            println("             |               Robust")
        end
        se_label = vce == :robust ? "std. err." : "Std. err."
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), se_label, 100*level)
        println("-"^13, "+", "-"^64)
        # Body: exog_vars..., endog, _cons
        for (i, lab) in enumerate(vcat([string(v) for v in exog_vars],
                                       string(endog), "_cons"))
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           lab, g9(β_disp[i]; w=10), g9(se_disp[i]; w=9),
                           Printf.@sprintf("%7.2f", z_disp[i]),
                           Printf.@sprintf("%.3f", p_disp[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^13, "+", "-"^64)
        # Auxiliary block: corr(e.linc, e.ins), sd(e.linc)
        Printf.@printf("%12s |\n", " corr(e.$endog,")
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "       e.$depvar)",
                       g9(ρ; w=10), g9(se_ρ; w=9), "",
                       g9(ρ_lo; w=9), g9(ρ_hi; w=10))
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "  sd(e.$endog)",
                       g9(σ_v; w=10), g9(se_σv; w=9), "",
                       g9(σv_lo; w=9), g9(σv_hi; w=10))
        println("-"^78)
        Printf.@printf("Wald test of exogeneity (corr = 0): chi2(1) = %.2f%8sProb > chi2 = %.4f\n",
                       Wald_exog, "", Wald_exog_p)
        println("Endogenous: ", string(endog))
        exo_full = vcat([string(v) for v in exog_vars],
                        [string(v) for v in instruments])
        # Wrap at ~12 chars after "Exogenous: " to mimic Stata's line wrap.
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

    return (; β = β_disp, se = se_disp, V = V_disp,
              γ = β_disp[length(exog_vars) + 1],
              π = π̂, σ_v, ρ, ln_σv, at_ρ,
              se_σv, se_ρ, σv_lo, σv_hi, ρ_lo, ρ_hi,
              ll, n, Wald, Wald_p, Wald_exog, Wald_exog_p,
              coefnames_struct = cn_struct,
              coefnames_first  = cn_first,
              model = m_pr, V_full)
end
