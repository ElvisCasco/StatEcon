# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_hetprob — `hetprob y x, het(z)` heteroskedastic probit (MLE)
# --------------------------------------------------------------------------
import Optim
import ForwardDiff

"""
    stata_hetprob(df, formula_mean, het_vars; level=0.95, quiet=false)
        -> NamedTuple

Stata-style `hetprob <depvar> <indeps>, het(<het_vars>) nolog`.
Heteroskedastic probit:

    y_i = 1{x_iᵀβ + ε_i > 0},   ε_i ~ N(0, σ_i²),   σ_i = exp(z_iᵀγ)

The variance specification carries no constant (absorbed into β's scale), so
`γ` has length `length(het_vars)`. Estimation by MLE (LBFGS with an explicit
ForwardDiff gradient), warm-started from a plain probit. OIM vcov from a
finite-difference Hessian of the negative log-likelihood.

Output mirrors Stata: header (Number of obs / Zero / Nonzero outcomes / Wald
chi2(k_β-slopes) / Log likelihood / Prob > chi2), a two-block coefficient
table (`<depvar>` block + `lnsigma` block), and the `LR test of lnsigma=0:
chi2(q)` line.

`het_vars` is a `Vector{Symbol}` (or anything convertible). Returns
`(; β, γ, se_β, se_γ, V, ll, ll_probit, LR, LR_p, Wald, Wald_p, n, n_zero,
n_one, k_x, k_z, coefnames_x, coefnames_z, X, Z, model)`.
"""
function stata_hetprob(df, formula_mean, het_vars; level::Float64 = 0.95,
                       quiet::Bool = false)
    het_syms = Symbol.(het_vars)
    needed   = unique(vcat(StatsModels.termvars(formula_mean), het_syms))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    # Build X (mean equation, with intercept) and Z (variance equation, no
    # intercept — Stata's hetprob convention).
    m_pr = GLM.glm(formula_mean, dfc, Distributions.Binomial(), GLM.ProbitLink())
    β0   = GLM.coef(m_pr); cn_x = GLM.coefnames(m_pr)
    Xm   = GLM.modelmatrix(m_pr)
    yv   = Float64.(GLM.response(m_pr))
    Z    = hcat([Float64.(dfc[!, v]) for v in het_syms]...)
    cn_z = String.(het_syms)
    n    = size(Xm, 1)
    k_x  = size(Xm, 2)
    k_z  = size(Z, 2)

    # Negative log-likelihood (jointly in (β, γ))
    function negll(θ)
        β = θ[1:k_x]; γ = θ[k_x+1:end]
        σ = exp.(Z * γ)
        s = (2 .* yv .- 1.0) .* (Xm * β) ./ σ
        return -sum(log.(max.(Distributions.cdf.(Distributions.Normal(), s),
                              1e-300)))
    end
    θ0  = vcat(β0, zeros(k_z))
    # Optim's `autodiff = :forward` kwarg pipeline is broken in the installed
    # Optim; wrap `negll` with an explicit ForwardDiff gradient and use the
    # 5-arg optimize form.
    g!(G, x) = ForwardDiff.gradient!(G, negll, x)
    res = Optim.optimize(negll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 1000))
    θ̂   = Optim.minimizer(res)
    β̂   = θ̂[1:k_x]
    γ̂   = θ̂[k_x+1:end]
    ll  = -negll(θ̂)
    ll_probit = GLM.loglikelihood(m_pr)
    LR  = 2 * (ll - ll_probit)
    LR_p = 1 - Distributions.cdf(Distributions.Chisq(k_z), LR)

    # OIM vcov via finite-difference Hessian (avoids ForwardDiff through
    # CDF a second time).
    function _fd_hessian(f, x)
        n_p = length(x); H = zeros(n_p, n_p)
        h   = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0  = f(x)
        for i in 1:n_p, j in i:n_p
            xpp = copy(x); xpp[i] += h[i]; xpp[j] += h[j]
            xmm = copy(x); xmm[i] -= h[i]; xmm[j] -= h[j]
            if i == j
                xp = copy(x); xp[i] += h[i]
                xm = copy(x); xm[i] -= h[i]
                H[i, i] = (f(xp) - 2*f0 + f(xm)) / h[i]^2
            else
                xpm = copy(x); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = copy(x); xmp[i] -= h[i]; xmp[j] += h[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
            end
        end
        return H
    end
    H = _fd_hessian(negll, θ̂)
    V = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))
    se_full = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β = se_full[1:k_x]
    se_γ = se_full[k_x+1:end]
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)

    # Wald chi² on β slopes (excluding intercept)
    intercept_idx = findfirst(==("(Intercept)"), cn_x)
    slope_x = setdiff(1:k_x, [intercept_idx])
    Wald, Wald_p = NaN, NaN
    if !isempty(slope_x)
        β_s   = β̂[slope_x]
        V_s   = V[slope_x, slope_x]
        Wald  = β_s' * LinearAlgebra.inv(V_s) * β_s
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_x)), Wald)
    end

    # Stata `%9.0g` formatter
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

    n_one  = sum(yv .== 1)
    n_zero = n - n_one

    # Print Stata-format output
    if !quiet
        println()
        # Header (right column auto-sized; ~8 chars wide for the values)
        nstr = commafmt(n)
        z0   = commafmt(n_zero)
        z1   = commafmt(n_one)
        wstr = isfinite(Wald) ? Printf.@sprintf("%.2f", Wald) : ""
        pstr = isfinite(Wald) ? Printf.@sprintf("%.4f", Wald_p) : ""
        vw   = maximum(length, (nstr, z0, z1, wstr, pstr))
        Printf.@printf("%-48s%-18s= %*s\n",
                       "Heteroskedastic probit model", "Number of obs", vw, nstr)
        Printf.@printf("%48s%-18s= %*s\n", "", "Zero outcomes", vw, z0)
        Printf.@printf("%48s%-18s= %*s\n", "", "Nonzero outcomes", vw, z1)
        println()
        Printf.@printf("%48s%-18s= %*s\n", "",
                       "Wald chi2($(length(slope_x)))", vw, wstr)
        ll_str = Printf.@sprintf("Log likelihood = %.3f", ll)
        right  = Printf.@sprintf("%-18s= %*s", "Prob > chi2", vw, pstr)
        pad_h  = max(0, 78 - length(ll_str) - length(right))
        println(ll_str, " "^pad_h, right)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(formula_mean.lhs), 100*level)
        println("-"^13, "+", "-"^64)

        # ins (mean) block — equation header, then slopes, then _cons
        println(rpad(string(formula_mean.lhs), 12), " |")
        z_β   = β̂ ./ se_β
        p_β   = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_β)))
        ci_lo_β = β̂ .- crit .* se_β
        ci_hi_β = β̂ .+ crit .* se_β
        ord_x = intercept_idx === nothing ? collect(slope_x) :
                                            vcat(slope_x, [intercept_idx])
        for i in ord_x
            label = cn_x[i] == "(Intercept)" ? "_cons" : cn_x[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(β̂[i]; w=10), g9(se_β[i]; w=9),
                           Printf.@sprintf("%7.2f", z_β[i]),
                           Printf.@sprintf("%.3f", p_β[i]),
                           g9(ci_lo_β[i]; w=9), g9(ci_hi_β[i]; w=10))
        end
        println("-"^13, "+", "-"^64)

        # lnsigma block
        println(rpad("lnsigma", 12), " |")
        z_γ   = γ̂ ./ se_γ
        p_γ   = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z_γ)))
        ci_lo_γ = γ̂ .- crit .* se_γ
        ci_hi_γ = γ̂ .+ crit .* se_γ
        for i in 1:k_z
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           cn_z[i], g9(γ̂[i]; w=10), g9(se_γ[i]; w=9),
                           Printf.@sprintf("%7.2f", z_γ[i]),
                           Printf.@sprintf("%.3f", p_γ[i]),
                           g9(ci_lo_γ[i]; w=9), g9(ci_hi_γ[i]; w=10))
        end
        println("-"^78)

        # LR test footer
        Printf.@printf("LR test of lnsigma=0: chi2(%d) = %.2f%26sProb > chi2 = %.4f\n",
                       k_z, LR, "", LR_p)
    end

    return (; β = β̂, γ = γ̂, se_β, se_γ, V, ll, ll_probit, LR, LR_p,
              Wald, Wald_p, n, n_zero, n_one, k_x, k_z,
              coefnames_x = cn_x, coefnames_z = cn_z,
              X = Xm, Z, model = m_pr)
end
