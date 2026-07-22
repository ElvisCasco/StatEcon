# ============================================================================
# stata_nbreg.jl — Stata `nbreg` negative-binomial (Cameron & Trivedi ch12)
#
# CANONICAL negative-binomial estimator (NB2). Later chapters reuse it.
#
# SpecialFunctions is NOT a StatEcon dependency, so the log-gamma / digamma
# functions the NB log-likelihood needs are implemented here dependency-free:
#   _c12_loggamma — Lanczos g=7 approximation (with reflection for x < 0.5)
#   _c12_digamma  — recurrence + asymptotic series
# The model is fit jointly in (β, lnα) by Optim.NelderMead (derivative-free,
# so no autodiff is pushed through the gamma functions) with the OIM vcov from
# a central finite-difference Hessian — the same validated numerics as the
# source, with the SpecialFunctions calls swapped for the local helpers.
# ============================================================================

import Optim

# ---------------------------------------------------------------------------
# Dependency-free log-gamma and digamma (accurate to ~1e-8 for x > 0, which
# covers every argument the NB2 log-likelihood evaluates: 1/α, y+1/α, y+1).
# ---------------------------------------------------------------------------
function _c12_loggamma(x::Real)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(π / sin(π * x)) - _c12_loggamma(1 - x)   # reflection
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 1:8
        a += c[i + 1] / (x + i)
    end
    return 0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end

function _c12_digamma(x::Real)
    ψ = zero(float(x))
    while x < 6                         # recurrence to push argument up
        ψ -= 1 / x
        x += 1
    end
    inv  = 1 / x
    inv2 = inv * inv
    ψ += log(x) - 0.5 * inv -
         inv2 * (1/12 - inv2 * (1/120 - inv2 * (1/252)))
    return ψ
end

# ---------------------------------------------------------------------------
# Lightweight fitted-model wrapper so StatsBase.coef / vcov / coefnames work
# on the returned `model` (coefnames in GLM order, `(Intercept)` labelled).
# ---------------------------------------------------------------------------
struct NBregModel
    β::Vector{Float64}
    V::Matrix{Float64}
    coefnames::Vector{String}
end
StatsBase.coef(m::NBregModel)      = m.β
StatsBase.vcov(m::NBregModel)      = m.V
StatsBase.coefnames(m::NBregModel) = m.coefnames

"""
    stata_nbreg(df, formula; vce=:oim, level=0.95, dispersion=:mean,
                quiet=false, cluster_var=nothing) -> NamedTuple

Stata-style `nbreg <depvar> <regs>, [vce(robust)] nolog`. Fits the NB2 model
(`dispersion=:mean`; the only variance function implemented) jointly in
`(β, lnα)` by maximising the log-likelihood

    lnf_i = lnΓ(y_i + 1/α) − lnΓ(1/α) − lnΓ(y_i + 1)
            − (y_i + 1/α)·ln(1 + α·μ_i) + y_i·ln(α) + y_i·θ_i,
    θ_i = xᵢᵀβ,  μ_i = exp(θ_i),

via `Optim.NelderMead` (warm-started from a Poisson fit) and uses the inverse
finite-difference Hessian as the OIM vcov. `vce(:robust)` / `vce(:cluster)`
build the sandwich vcov from finite-difference per-observation scores.

Prints the Stata header (Number of obs / LR|Wald chi2(k) / Prob > chi2 /
Pseudo R²) followed by the coefficient table with `/lnalpha` and `alpha` rows,
and (for `vce(:oim)`) the LR test of `α = 0` (NB → Poisson) reported as
`chibar2(01)` with the half-mixture p-value `0.5·P(χ²₁ > LR)`.

Returns a NamedTuple with fields (at least):
`(; model, β, se, V, coefnames, ll, n, alpha, lnalpha)` where `β, se, V,
coefnames` are in Stata display order (slopes first, `coefnames` ending in
`"_cons"`). Also exposes GLM-order copies (`β_glm, se_glm, V_glm,
coefnames_glm`), `se_lnalpha, se_alpha, ll_pois, ll_null, pseudo_r2, k, LR,
LR_p, LR_alpha, p_chibar`.
"""
function stata_nbreg(df, formula; vce::Symbol=:oim, level::Float64=0.95,
                    dispersion::Symbol=:mean,
                    quiet::Bool=false, cluster_var=nothing)
    dispersion == :mean ||
        error("stata_nbreg: only dispersion=:mean (NB2) is implemented.")
    needed = StatsModels.termvars(formula)
    cols   = cluster_var === nothing ? needed :
                                       vcat(needed, [Symbol(cluster_var)])
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    # Fit Poisson first: gives β starting values + LL_pois for the LR test.
    m_pois  = GLM.glm(formula, dfc, Distributions.Poisson(), GLM.LogLink())
    β0      = GLM.coef(m_pois)
    ll_pois = GLM.loglikelihood(m_pois)
    yv      = Float64.(GLM.response(m_pois))
    Xm      = GLM.modelmatrix(m_pois)
    cn      = GLM.coefnames(m_pois)
    n       = size(Xm, 1)
    k       = size(Xm, 2)

    # NB log-likelihood (parameters are β stacked with lnα — last entry).
    function nb_nll(p)
        β  = p[1:k]; α = exp(p[k + 1])
        η  = Xm * β; μ = exp.(η); inv_α = 1 / α
        ll = sum(_c12_loggamma.(yv .+ inv_α) .-
                 _c12_loggamma(inv_α) .-
                 _c12_loggamma.(yv .+ 1) .-
                 (yv .+ inv_α) .* log.(1 .+ α .* μ) .+
                 yv .* log(α) .+ yv .* η)
        return -ll
    end
    p0  = Float64.(vcat(β0, log(1.0)))
    res = Optim.optimize(nb_nll, p0, Optim.NelderMead(),
                         Optim.Options(g_tol      = 1e-10,
                                       x_abstol   = 1e-10,
                                       f_reltol   = 1e-12,
                                       iterations = 5000))
    p̂   = Float64.(Optim.minimizer(res))
    β̂   = p̂[1:k]
    lnα = p̂[k + 1]
    α   = exp(lnα)
    ll  = -nb_nll(p̂)::Float64

    # OIM vcov from inverse Hessian via central finite differences.
    function _fd_hessian(f::Function, x::AbstractVector{<:Real})
        n_p = length(x)
        H   = zeros(n_p, n_p)
        h   = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0  = f(x)
        for i in 1:n_p
            xpi = copy(x); xmi = copy(x)
            xpi[i] += h[i]; xmi[i] -= h[i]
            fpi, fmi = f(xpi), f(xmi)
            H[i, i] = (fpi - 2*f0 + fmi) / h[i]^2
            for j in (i+1):n_p
                xpp = copy(x); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = copy(x); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = copy(x); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = copy(x); xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
            end
        end
        return H
    end
    H = _fd_hessian(nb_nll, p̂)
    V_oim = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))

    # Sandwich vcov for robust / cluster; per-obs scores by finite differences.
    V = if vce == :robust || vce == :cluster
        function nb_nll_i(p, i)
            β_l = p[1:k]; α_l = exp(p[k + 1])
            η_i  = LinearAlgebra.dot(view(Xm, i, :), β_l)
            μ_i  = exp(η_i); inv_α = 1 / α_l
            ll_i = _c12_loggamma(yv[i] + inv_α) -
                   _c12_loggamma(inv_α) -
                   _c12_loggamma(yv[i] + 1) -
                   (yv[i] + inv_α) * log(1 + α_l * μ_i) +
                   yv[i] * log(α_l) + yv[i] * η_i
            return -ll_i
        end
        h_g  = sqrt(sqrt(eps(Float64))) .* max.(abs.(p̂), 1.0)
        S = zeros(n, k + 1)
        for i in 1:n, j in 1:(k + 1)
            xp = copy(p̂); xp[j] += h_g[j]
            xm = copy(p̂); xm[j] -= h_g[j]
            S[i, j] = (nb_nll_i(xp, i) - nb_nll_i(xm, i)) / (2 * h_g[j])
        end
        if vce == :robust
            B = S' * S
            V_oim * B * V_oim * (n / (n - 1))
        else
            cluster_var === nothing &&
                error("vce = :cluster requires `cluster_var = :varname`")
            cl = dfc[!, Symbol(cluster_var)]
            B  = zeros(k + 1, k + 1)
            for g in unique(cl)
                sg = vec(sum(S[cl .== g, :], dims = 1))
                B .+= sg * sg'
            end
            G = length(unique(cl))
            V_oim * B * V_oim * (G / (G - 1))
        end
    else
        V_oim
    end
    se_full = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β    = se_full[1:k]
    se_lnα  = se_full[k + 1]
    se_α    = α * se_lnα                   # delta method on exp(lnα)

    crit  = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    z_β   = β̂ ./ se_β
    p_β   = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_β)))
    ci_lo_β = β̂ .- crit .* se_β
    ci_hi_β = β̂ .+ crit .* se_β
    lnα_lo, lnα_hi = lnα - crit*se_lnα, lnα + crit*se_lnα
    α_lo,   α_hi   = exp(lnα_lo),       exp(lnα_hi)

    # Null (intercept-only NB) for pseudo R². β̂_0 = log(mean y); α via 1D max.
    ȳ       = Statistics.mean(yv)
    β0_null = log(ȳ)
    function nb_ll_null_α(α::Float64)
        μ = fill(ȳ, length(yv))
        inv_α = 1 / α
        sum(_c12_loggamma.(yv .+ inv_α) .-
            _c12_loggamma(inv_α) .-
            _c12_loggamma.(yv .+ 1) .-
            (yv .+ inv_α) .* log.(1 .+ α .* μ) .+
            yv .* log(α) .+ yv .* β0_null)
    end
    res0 = Optim.optimize(α -> -nb_ll_null_α(exp(α[1])), [log(1.0)],
                          Optim.NelderMead())
    α̂_null   = exp(Float64(Optim.minimizer(res0)[1]))
    ll_null  = nb_ll_null_α(α̂_null)
    pseudo_r2 = 1 - ll / ll_null

    # LR chi2 (model vs intercept only) — NB likelihood ratio.
    LR    = 2 * (ll - ll_null)
    LR_p  = 1 - Distributions.cdf(Distributions.Chisq(k - 1), LR)

    # LR for α = 0 (NB vs Poisson). Half-mixture p-value: 0.5·P(χ²₁ > LR).
    LR_α  = 2 * (ll - ll_pois)
    p_chibar = LR_α <= 0 ? 1.0 :
               0.5 * Distributions.ccdf(Distributions.Chisq(1), LR_α)

    # Stata `%9.0g` formatter (sign-aware width cap).
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
    function commafmt(num)
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    # Wald chi² on slopes (used in robust header instead of LR).
    intercept_idx = findfirst(==("(Intercept)"), cn)
    slope_idx     = setdiff(1:k, [intercept_idx])
    Wald, Wald_p = NaN, NaN
    if !isempty(slope_idx)
        β_s = β̂[slope_idx]; V_s = V[slope_idx, slope_idx]
        Wald   = β_s' * LinearAlgebra.inv(V_s) * β_s
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                       Wald)
    end

    if !quiet
        println()
        is_rob = vce == :robust || vce == :cluster
        nstr = commafmt(n)
        chi2_label = is_rob ? "Wald" : "LR"
        chi2_val   = is_rob ? Wald   : LR
        chi2_p     = is_rob ? Wald_p : LR_p
        wstr = Printf.@sprintf("%.2f", chi2_val)
        pstr = Printf.@sprintf("%.4f", chi2_p)
        rstr = Printf.@sprintf("%.4f", pseudo_r2)
        vw   = maximum(length, (nstr, wstr, pstr, rstr))
        Printf.@printf("%-56s%-13s = %*s\n",
                       "Negative binomial regression", "Number of obs", vw, nstr)
        Printf.@printf("%56s%-13s = %*s\n", "",
                       "$chi2_label chi2($(length(slope_idx)))", vw, wstr)
        Printf.@printf("%-56s%-13s = %*s\n",
                       "Dispersion: mean", "Prob > chi2", vw, pstr)
        ll_label = is_rob ? "Log pseudolikelihood" : "Log likelihood"
        Printf.@printf("%-56s%-13s = %*s\n",
                       Printf.@sprintf("%s = %.4f", ll_label, ll),
                       "Pseudo R2", vw, rstr)
        println()

        if vce == :cluster
            G = length(unique(dfc[!, Symbol(cluster_var)]))
            cl_str = Printf.@sprintf("(Std. err. adjusted for %s clusters in %s)",
                                     commafmt(G), cluster_var)
            println(lpad(cl_str, 78))
        end

        ord = intercept_idx === nothing ? collect(slope_idx) :
                                          vcat(slope_idx, [intercept_idx])
        println("-"^78)
        if is_rob
            println("             |               Robust")
        end
        se_label = is_rob ? "std. err." : "Std. err."
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       string(formula.lhs), se_label, 100*level)
        println("-"^13, "+", "-"^64)
        for i in ord
            label = cn[i] == "(Intercept)" ? "_cons" : cn[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(β̂[i]; w=10), g9(se_β[i]; w=9),
                           Printf.@sprintf("%7.2f", z_β[i]),
                           Printf.@sprintf("%.3f", p_β[i]),
                           g9(ci_lo_β[i]; w=9), g9(ci_hi_β[i]; w=10))
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "/lnalpha", g9(lnα; w=10), g9(se_lnα; w=9), "",
                       g9(lnα_lo; w=9), g9(lnα_hi; w=10))
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %s  %s%26s%s  %s\n",
                       "alpha", g9(α; w=10), g9(se_α; w=9), "",
                       g9(α_lo; w=9), g9(α_hi; w=10))
        println("-"^78)
        if !is_rob
            chibar_str = LR_α >= 1e4 ? Printf.@sprintf("%.1e", LR_α) :
                         Printf.@sprintf("%.2f", LR_α)
            Printf.@printf("LR test of alpha=0: chibar2(01) = %-15s Prob >= chibar2 = %.3f\n",
                           chibar_str, p_chibar)
        end
    end

    # Stata display order: slopes first, then _cons.
    ord    = intercept_idx === nothing ? collect(slope_idx) :
                                         vcat(slope_idx, [intercept_idx])
    cn_ord = intercept_idx === nothing ? cn[ord] :
                                         vcat(cn[slope_idx], "_cons")
    V_ββ   = V[1:k, 1:k]

    return (; model = NBregModel(β̂, V_ββ, cn),
              β = β̂[ord], se = se_β[ord], V = V_ββ[ord, ord],
              coefnames = cn_ord,
              β_glm = β̂, se_glm = se_β, V_glm = V_ββ, coefnames_glm = cn,
              lnalpha = lnα, se_lnalpha = se_lnα, alpha = α, se_alpha = se_α,
              ll, ll_pois, ll_null, pseudo_r2,
              n, k, LR, LR_p, LR_alpha = LR_α, p_chibar)
end
