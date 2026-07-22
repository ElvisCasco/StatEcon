# ============================================================================
# stata_gnbreg.jl — Stata `gnbreg` generalized NB2 (Cameron & Trivedi ch17)
#
# The NB2 log-likelihood's log-gamma terms use the ch17-shared `_c17_loggamma`
# (defined in stata_ztnb.jl). Because `_c17_loggamma` is pure arithmetic it is
# ForwardDiff-compatible, so the joint (β, γ) fit keeps the source's explicit
# ForwardDiff gradient with the 5-arg `Optim.optimize(f, g!, x0, method, opts)`
# form (the source's `autodiff=:forward` kwarg path is broken in this Optim).
# ============================================================================

import Optim
import ForwardDiff

"""
    stata_gnbreg(df, formula; lnalpha_vars=Symbol[], level=0.95, quiet=false)

Stata's `gnbreg y x..., lnalpha(z1 z2 ...) nolog` — generalized NB2 with
the dispersion parameter modelled as ln(α_i) = z_i'γ. Jointly MLE
estimates β (mean equation, μ = exp(x'β)) and γ (variance equation,
α = exp(z'γ)) by LBFGS + ForwardDiff, with OIM SEs from a finite-
difference Hessian. The Stata-format printout reproduces the
two-equation block (outcome panel and `lnalpha` variance
panel), pseudo-R² against the Poisson-restricted model, and a Wald
χ²(k_x) for joint slope significance.

Returns a NamedTuple `(; β, γ, V, se_β, se_γ, ll, ll_pois, ll_null,
pseudo_r2, n, k_x, k_z, coefnames_x, coefnames_z, Wald, Wald_p)`.

`gnbreg` is notorious for non-convergence on real data (Cameron &
Trivedi flag this); the caller may need to wrap in a `try / catch`
analogous to Stata's `capture noisily gnbreg …`.
"""
function stata_gnbreg(df, formula;
                     lnalpha_vars::AbstractVector{Symbol} = Symbol[],
                     level::Float64 = 0.95, quiet::Bool = false)
    needed = unique(vcat(StatsModels.termvars(formula),
                         collect(lnalpha_vars)))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    # Warm-start β from Poisson (Stata does too); μ = exp(x'β).
    m_pois  = GLM.glm(formula, dfc, Distributions.Poisson(), GLM.LogLink())
    β0      = GLM.coef(m_pois)
    Xm      = GLM.modelmatrix(m_pois)
    yv      = Float64.(GLM.response(m_pois))
    cn_X    = GLM.coefnames(m_pois)
    n       = size(Xm, 1)
    k_x     = size(Xm, 2)
    ll_pois = GLM.loglikelihood(m_pois)

    # Z matrix for ln(α). Stata's `gnbreg` prints `lnalpha` rows in
    # the order [<covariates in lnalpha(…) order>, _cons], so we put
    # covariates first and the constant last — that way γ̂ entries
    # line up with `cn_Z` without any per-row reordering at print
    # time.
    Zm   = hcat([Float64.(_c17_rawval.(dfc[!, v])) for v in lnalpha_vars]...,
                ones(n))
    cn_Z = vcat(string.(lnalpha_vars), "_cons")
    k_z  = size(Zm, 2)

    function nll(θ)
        β = view(θ, 1:k_x); γ = view(θ, (k_x + 1):(k_x + k_z))
        η = Xm * β; μ = exp.(η)
        ζ = Zm * γ; α = exp.(ζ); iα = 1 ./ α
        ll = zero(eltype(θ))
        for i in 1:n
            yi = yv[i]; μi = μ[i]; αi = α[i]; iαi = iα[i]
            ll += _c17_loggamma(yi + iαi) -
                  _c17_loggamma(iαi) -
                  _c17_loggamma(yi + 1) -
                  (yi + iαi) * log(1 + αi * μi) +
                  yi * log(αi) + yi * log(μi)
        end
        return -ll
    end

    θ0 = vcat(β0, zeros(k_z))
    g! = (g, x) -> ForwardDiff.gradient!(g, nll, x)
    res = Optim.optimize(nll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-7, iterations = 500))
    θ̂  = Optim.minimizer(res)
    β̂  = θ̂[1:k_x]
    γ̂  = θ̂[(k_x + 1):(k_x + k_z)]
    ll = -nll(θ̂)

    # OIM SEs via finite-difference Hessian.
    function _fd_hess(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x); xpi[i] += h_[i]; xmi[i] -= h_[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h_[i]^2
            for j in (i + 1):nθ
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
    V    = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hess(nll, θ̂)))
    se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β = se[1:k_x]
    se_γ = se[(k_x + 1):(k_x + k_z)]

    yname = string(formula.lhs)

    # Stata %9.0g formatter
    function g9(x; w::Int = 10, sig::Int = 7)
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

    # Wald χ²(k_slopes) on β (excluding the outcome's intercept)
    cons_idx = findfirst(==("(Intercept)"), cn_X)
    slope_idx = setdiff(1:k_x, [cons_idx])
    Wald = NaN; Wald_p = NaN
    if !isempty(slope_idx)
        V_sl = V[slope_idx, slope_idx]
        Wald = β̂[slope_idx]' * LinearAlgebra.inv(V_sl) * β̂[slope_idx]
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                       Wald)
    end

    # ll_null: Poisson on the intercept alone (slopes set to 0; standard
    # null for the pseudo-R² display).
    m_null = GLM.glm(StatsModels.term(Symbol(formula.lhs)) ~
                      StatsModels.term(1),
                     dfc, Distributions.Poisson(), GLM.LogLink())
    ll_null = GLM.loglikelihood(m_null)
    pseudo_r2 = 1 - ll / ll_null

    crit = Distributions.quantile(Distributions.Normal(),
                                  1 - (1 - level) / 2)

    if !quiet
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Generalized negative binomial regression",
                       "Number of obs", commafmt(n))
        if isfinite(Wald)
            Printf.@printf("%56s%-13s = %6.2f\n", "",
                           "Wald chi2($(length(slope_idx)))", Wald)
            Printf.@printf("%56s%-13s = %6.4f\n", "",
                           "Prob > chi2", Wald_p)
        end
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad_h  = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad_h, r2_str)
        println()

        # Coefficient table: outcome equation then lnalpha equation.
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [95%% conf. interval]\n",
                       yname)
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", yname)
        function _row(label, b, s)
            z = b / s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit * s; hi = b + crit * s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        # Outcome equation: slopes first, then _cons.
        ord_x = vcat(slope_idx, cons_idx === nothing ? Int[] : [cons_idx])
        for i in ord_x
            lab = cn_X[i] == "(Intercept)" ? "_cons" : cn_X[i]
            _row(lab, β̂[i], se_β[i])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", "lnalpha")
        # lnalpha equation: covariates first, then _cons.
        for j in 1:k_z
            _row(cn_Z[j], γ̂[j], se_γ[j])
        end
        println("-"^78)
    end

    return (; β = β̂, γ = γ̂, V, se_β, se_γ,
              ll, ll_pois, ll_null, pseudo_r2,
              n, k_x, k_z,
              coefnames_x = cn_X, coefnames_z = cn_Z,
              Wald, Wald_p)
end
