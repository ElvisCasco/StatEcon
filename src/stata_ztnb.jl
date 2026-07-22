# ============================================================================
# stata_ztnb.jl — Stata `ztnb` zero-truncated NB2 (Cameron & Trivedi ch17)
#
# PRIMARY ch17 count-model file. Defines the two chapter-shared helpers used
# across the ch17 count-data commands (runtime cross-file references are fine
# because every ch17 file is `include`d into the same `module StatEcon`):
#
#   _c17_rawval   — unwrap a ReadStatTables.LabeledValue to its raw number
#   _c17_loggamma — dependency-free log-Γ (Lanczos g=7, reflection for x<0.5)
#
# SpecialFunctions is NOT a StatEcon dependency, so the NB2 log-likelihood's
# log-gamma terms use `_c17_loggamma`. It is pure arithmetic (+ log/sin), so it
# is ForwardDiff-compatible (needed by `stata_gnbreg`, which differentiates its
# NB nll). No ch17 command needs digamma (all vcovs come from finite-difference
# Hessians/scores), so only log-gamma is provided.
# ============================================================================

import Optim

# Unwrap a ReadStatTables.LabeledValue (has `.value`); pass plain reals through.
_c17_rawval(x) = hasproperty(x, :value) ? x.value : x

# Dependency-free log-Γ (Lanczos g=7). Accurate to ~1e-8 for the arguments the
# ch17 count likelihoods evaluate (1/α, y+1/α, y+1). ForwardDiff-safe.
function _c17_loggamma(x::Real)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(π / sin(π * x)) - _c17_loggamma(1 - x)   # reflection
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 1:8
        a += c[i + 1] / (x + i)
    end
    return 0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end

"""
    stata_ztnb(df, formula; vce=:oim, level=0.95, quiet=false) -> NamedTuple

Stata's `ztnb <depvar> <regs> [if depvar>0], nolog` — zero-truncated
NB2, the count part of a hurdle (or on-site-sampling) model. Same NB2
kernel as `stata_nbreg` but each observation's density is renormalised
by the positive-count probability:

    lnf_i = lnf_i^{NB2} − ln(1 − P(Y_i = 0)),
    P(Y_i = 0) = (1 + α·μ_i)^{−1/α},  μ_i = exp(x_i'β)

The caller is responsible for passing only the positive subsample (the
`if depvar>0` qualifier) — the helper does not filter. Jointly MLE in
(β, lnα) by NelderMead (gamma-safe, no autodiff), OIM vcov from a
central-difference Hessian, optional robust sandwich. Prints the Stata
`Zero-truncated negative binomial regression` block (header, coefficient
table, `/lnalpha` + `alpha` rows).

Returns `(; β, V, se_β, lnα, se_lnα, α, se_α, ll, n, k, coefnames, X)`.
"""
function stata_ztnb(df, formula; vce::Symbol = :oim, level::Float64 = 0.95,
                   quiet::Bool = false)
    needed = StatsModels.termvars(formula)
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    # Poisson warm start (β only).
    m_pois = GLM.glm(formula, dfc, Distributions.Poisson(), GLM.LogLink())
    β0  = GLM.coef(m_pois)
    yv  = Float64.(GLM.response(m_pois))
    Xm  = GLM.modelmatrix(m_pois)
    cn  = GLM.coefnames(m_pois)
    n   = size(Xm, 1); k = size(Xm, 2)

    # ZT-NB2 negative log-likelihood (β stacked with lnα).
    function ztnb_nll(p)
        β = p[1:k]; α = exp(p[k + 1]); iα = 1 / α
        η = Xm * β; μ = exp.(η)
        ll = 0.0
        for i in 1:n
            μi = μ[i]; yi = yv[i]
            lnf = _c17_loggamma(yi + iα) -
                  _c17_loggamma(iα) -
                  _c17_loggamma(yi + 1) -
                  (yi + iα) * log(1 + α * μi) +
                  yi * log(α) + yi * η[i]
            p0  = (1 + α * μi)^(-iα)                 # P(Y = 0)
            ll += lnf - log(max(1 - p0, eps()))      # zero-truncation
        end
        return -ll
    end

    p0_start = Float64.(vcat(β0, log(1.0)))
    res = Optim.optimize(ztnb_nll, p0_start, Optim.NelderMead(),
                         Optim.Options(g_tol = 1e-10, x_abstol = 1e-10,
                                       f_reltol = 1e-12, iterations = 5000))
    p̂   = Float64.(Optim.minimizer(res))
    β̂   = p̂[1:k]; lnα = p̂[k + 1]; α = exp(lnα)
    ll  = -ztnb_nll(p̂)::Float64

    # OIM Hessian via central finite differences (gamma-safe).
    function _fd_hessian(f, x)
        np = length(x); H = zeros(np, np)
        h  = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:np
            xpi = copy(x); xmi = copy(x); xpi[i] += h[i]; xmi[i] -= h[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h[i]^2
            for j in (i+1):np
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
    H     = _fd_hessian(ztnb_nll, p̂)
    V_oim = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))
    # Robust (Huber–White) sandwich: V = A⁻¹ B A⁻¹ · n/(n−1), A = OIM
    # Hessian, B = Σ_i s_i s_iᵀ. Per-obs scores via central differences
    # of the single-observation ZT-NB nll (gamma-safe).
    V = if vce == :robust
        function ztnb_nll_i(p, i)
            β = p[1:k]; α_l = exp(p[k + 1]); iα = 1 / α_l
            ηi = LinearAlgebra.dot(view(Xm, i, :), β); μi = exp(ηi)
            lnf = _c17_loggamma(yv[i] + iα) -
                  _c17_loggamma(iα) -
                  _c17_loggamma(yv[i] + 1) -
                  (yv[i] + iα) * log(1 + α_l * μi) +
                  yv[i] * log(α_l) + yv[i] * ηi
            p0 = (1 + α_l * μi)^(-iα)
            return -(lnf - log(max(1 - p0, eps())))
        end
        hg = sqrt(sqrt(eps(Float64))) .* max.(abs.(p̂), 1.0)
        S  = zeros(n, k + 1)
        for i in 1:n, j in 1:(k + 1)
            xp = copy(p̂); xp[j] += hg[j]
            xm = copy(p̂); xm[j] -= hg[j]
            S[i, j] = (ztnb_nll_i(xp, i) - ztnb_nll_i(xm, i)) / (2 * hg[j])
        end
        B = S' * S
        V_oim * B * V_oim * (n / (n - 1))
    else
        V_oim
    end
    se_full = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β  = se_full[1:k]; se_lnα = se_full[k + 1]; se_α = α * se_lnα

    crit  = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    z_β   = β̂ ./ se_β
    p_β   = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_β)))
    ci_lo_β = β̂ .- crit .* se_β; ci_hi_β = β̂ .+ crit .* se_β
    lnα_lo, lnα_hi = lnα - crit*se_lnα, lnα + crit*se_lnα
    α_lo, α_hi = exp(lnα_lo), exp(lnα_hi)

    intercept_idx = findfirst(==("(Intercept)"), cn)
    slope_idx = setdiff(1:k, intercept_idx === nothing ? Int[] : [intercept_idx])
    Wald = NaN; Wald_p = NaN
    if !isempty(slope_idx)
        V_sl = V[slope_idx, slope_idx]
        Wald = β̂[slope_idx]' * LinearAlgebra.inv(V_sl) * β̂[slope_idx]
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                       Wald)
    end

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        su = sig; s = Printf.@sprintf("%.*g", su, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && su > 1
            su -= 1; s = Printf.@sprintf("%.*g", su, x)
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
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Zero-truncated negative binomial regression",
                       "Number of obs", commafmt(n))
        if isfinite(Wald)
            Printf.@printf("%56s%-13s = %6.2f\n", "",
                           "LR chi2($(length(slope_idx)))", Wald)
            Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", Wald_p)
        end
        Printf.@printf("Log likelihood = %.4f\n\n", ll)

        ord = intercept_idx === nothing ? collect(slope_idx) :
                                          vcat(slope_idx, [intercept_idx])
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(formula.lhs), 100*level)
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
    end

    return (; β = β̂, V, se_β, lnα, se_lnα, α, se_α,
              ll, n, k, coefnames = cn, X = Xm)
end
