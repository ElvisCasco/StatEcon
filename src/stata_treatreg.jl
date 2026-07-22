# --------------------------------------------------------------------------
# stata_treatreg — Heckman treatment-effects model with endogenous binary
# treatment (Stata's `treatreg y indepvars, treat(D = varlist)`).
#
#   Outcome    : yᵢ = Xᵢβ + δDᵢ + εᵢ
#   Treatment  : Dᵢ = 𝟙{Zᵢγ + uᵢ > 0}
#   (ε, u) ~ 𝒩(0, [[σ² , ρσ], [ρσ, 1]])
#
# Full-information MLE via Optim.LBFGS with ForwardDiff Hessian for SEs;
# ρ, σ parameterized as `athrho = atanh(ρ)`, `lnsigma = log(σ)` for
# numerical stability. Requires `Optim` and `ForwardDiff` (added deps).
# --------------------------------------------------------------------------

import Optim
import ForwardDiff

# Optim autodiff compat shim for versions where `autodiff = :forward` is
# rejected by the underlying ADTypes/DifferentiationInterface plumbing.
# When `autodiff = :forward` is requested, build the gradient via
# ForwardDiff and route through the explicit-gradient 5-arg form.
const _TR_OPT_REAL = Optim.optimize
function _tr_optimize_compat(args...; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    ad = pop!(kw, :autodiff, nothing)
    if ad isa Symbol && (ad === :forward || ad === :forwarddiff) && length(args) >= 2
        f, x0 = args[1], args[2]
        g!(g, x) = ForwardDiff.gradient!(g, f, x)
        return _TR_OPT_REAL(f, g!, x0, args[3:end]...; kw...)
    end
    return ad === nothing ? _TR_OPT_REAL(args...; kw...) :
                            _TR_OPT_REAL(args...; autodiff = ad, kw...)
end

# Compact-format helper (Stata `%g`-ish).
_tr_g(x, w; sig = 7) = begin
    (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
    s = Printf.@sprintf("%.*g", sig, x)
    0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
    Printf.@sprintf("%*s", w, s)
end
_tr_fmtn(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")

"""
    stata_treatreg(df, y, exog, treat, treat_exog;
                   level=0.95, max_iter=2000, show_trace=false)

Stata-style `treatreg y indepvars, treat(D = varlist)` — Heckman
treatment-effects model, MLE. Returns a NamedTuple with `β, γ, ρ, σ, λ`
plus vcov and Wald/LR statistics. Prints the Stata block (summary
header → outcome → treatment → athrho/lnsigma → rho/sigma/lambda → LR
test of ρ = 0).
"""
function stata_treatreg(df, y, exog::AbstractVector, treat,
                        treat_exog::AbstractVector;
                        level::Float64 = 0.95, max_iter::Int = 2000,
                        show_trace::Bool = false)
    ys           = Symbol(y)
    exog_v       = [Symbol(v) for v in exog]
    treat_v      = Symbol(treat)
    treat_exog_v = [Symbol(v) for v in treat_exog]

    needed = unique(vcat(ys, exog_v, treat_v, treat_exog_v))
    d = DataFrames.dropmissing(df, needed)
    n = DataFrames.nrow(d)

    Y = Float64.(_sm_rawval.(d[!, ys]))
    D = Float64.(_sm_rawval.(d[!, treat_v]))
    X_out   = hcat([Float64.(_sm_rawval.(d[!, v])) for v in exog_v]...,
                    D, ones(n))
    X_treat = hcat([Float64.(_sm_rawval.(d[!, v])) for v in treat_exog_v]...,
                    ones(n))
    k_out   = size(X_out, 2)
    k_treat = size(X_treat, 2)

    names_out   = vcat(string.(exog_v), string(treat_v), "_cons")
    names_treat = vcat(string.(treat_exog_v), "_cons")

    function nll(θ)
        β      = θ[1:k_out]
        γ      = θ[k_out+1:k_out+k_treat]
        athrho = θ[k_out+k_treat+1]
        lnsig  = θ[k_out+k_treat+2]
        ρ = tanh(athrho); σ = exp(lnsig)
        ε = Y .- X_out * β
        η = X_treat * γ
        denom = sqrt(1 - ρ^2)
        ll = -n * (log(σ) + 0.5 * log(2π)) - 0.5 * sum((ε ./ σ) .^ 2)
        arg = (2 .* D .- 1) .* (η .+ ρ .* ε ./ σ) ./ denom
        ll += sum(log.(max.(Distributions.cdf.(Distributions.Normal(), arg),
                             1e-300)))
        return -ll
    end

    # Starts: OLS + probit + ρ=0, σ = OLS residual SD.
    β0 = X_out \ Y
    f_pr = term(treat_v) ~ sum(term.(treat_exog_v))
    pfit = GLM.glm(f_pr, d, GLM.Binomial(), GLM.ProbitLink())
    pcf  = GLM.coef(pfit); pnm = string.(GLM.coefnames(pfit))
    γ0   = zeros(k_treat)
    for (i, v) in enumerate(treat_exog_v)
        j = findfirst(==(string(v)), pnm)
        j !== nothing && (γ0[i] = pcf[j])
    end
    ji = findfirst(==("(Intercept)"), pnm)
    ji !== nothing && (γ0[end] = pcf[ji])
    ε0 = Y .- X_out * β0
    σ0 = sqrt(Statistics.mean(ε0 .^ 2))
    θ0 = vcat(β0, γ0, 0.0, log(σ0))

    res  = _tr_optimize_compat(nll, θ0, Optim.LBFGS(),
                               Optim.Options(iterations = max_iter,
                                             show_trace = show_trace);
                               autodiff = :forward)
    θ̂    = Optim.minimizer(res)
    logL = -Optim.minimum(res)
    H    = ForwardDiff.hessian(nll, θ̂)
    V    = LinearAlgebra.inv(H)
    se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    β̂ = θ̂[1:k_out]
    γ̂ = θ̂[k_out+1:k_out+k_treat]
    athrho = θ̂[k_out+k_treat+1]
    lnsig  = θ̂[k_out+k_treat+2]
    ρ̂ = tanh(athrho); σ̂ = exp(lnsig); λ̂ = ρ̂ * σ̂
    se_β   = se[1:k_out]
    se_γ   = se[k_out+1:k_out+k_treat]
    se_ath = se[k_out+k_treat+1]
    se_lns = se[k_out+k_treat+2]

    zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    V_ag  = V[[k_out+k_treat+1, k_out+k_treat+2],
              [k_out+k_treat+1, k_out+k_treat+2]]
    se_ρ  = sqrt(max((1 - ρ̂^2)^2 * V_ag[1, 1], 0.0))
    se_σ  = σ̂ * se_lns
    Jλ    = [σ̂ * (1 - ρ̂^2), ρ̂ * σ̂]
    se_λ  = sqrt(max(Jλ' * V_ag * Jλ, 0.0))

    ρ_ci = (tanh(athrho - zcrit*se_ath), tanh(athrho + zcrit*se_ath))
    σ_ci = (exp(lnsig - zcrit*se_lns),   exp(lnsig + zcrit*se_lns))
    λ_ci = (λ̂ - zcrit*se_λ,              λ̂ + zcrit*se_λ)

    # Wald χ² for outcome slopes (exclude _cons at last position).
    slope_idx = 1:(k_out-1)
    βs = β̂[slope_idx]; Vs = V[slope_idx, slope_idx]
    Wald = βs' * LinearAlgebra.inv(Vs) * βs
    df_wald = length(slope_idx)
    p_wald  = 1 - Distributions.cdf(Distributions.Chisq(df_wald), Wald)

    # LR test of ρ = 0 vs independent OLS + probit.
    σ²_r = Statistics.mean(ε0 .^ 2)
    logL_ols    = -n/2 * (log(2π) + log(σ²_r) + 1)
    logL_probit = GLM.loglikelihood(pfit)
    logL_r = logL_ols + logL_probit
    LR = -2 * (logL_r - logL)
    p_LR = 1 - Distributions.cdf(Distributions.Chisq(1), LR)

    # ── Stata-style output
    println()
    Printf.@printf("Treatment-effects model -- MLE%24s Number of obs = %7s\n",
                   "", _tr_fmtn(n))
    Printf.@printf("%54s Wald chi2(%d)  = %7.2f\n", "", df_wald, Wald)
    Printf.@printf("Log likelihood = %10.3f%28s Prob > chi2   = %7.4f\n",
                   logL, "", p_wald)
    println()

    println("-"^78)
    Printf.@printf("%12s | %-11s  %-9s  %6s  %5s     [%d%% conf. interval]\n",
                   "", "Coefficient", "Std. err.", "z", "P>|z|",
                   round(Int, 100*level))
    println("-"^13, "+", "-"^64)

    # Outcome block
    Printf.@printf("%-12s |\n", string(ys))
    for i in 1:k_out
        b = β̂[i]; s = se_β[i]
        z = b/s; p = 2*(1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        lo = b - zcrit*s; hi = b + zcrit*s
        Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                       names_out[i], _tr_g(b,10), _tr_g(s,9),
                       z, p, _tr_g(lo,11), _tr_g(hi,10))
    end
    println("-"^13, "+", "-"^64)

    # Treatment block
    Printf.@printf("%-12s |\n", string(treat_v))
    for i in 1:k_treat
        g = γ̂[i]; s = se_γ[i]
        z = g/s; p = 2*(1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        lo = g - zcrit*s; hi = g + zcrit*s
        Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                       names_treat[i], _tr_g(g,10), _tr_g(s,9),
                       z, p, _tr_g(lo,11), _tr_g(hi,10))
    end
    println("-"^13, "+", "-"^64)

    # Auxiliary
    for (nm, v, s) in (("/athrho", athrho, se_ath), ("/lnsigma", lnsig, se_lns))
        z = v/s; p = 2*(1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        lo = v - zcrit*s; hi = v + zcrit*s
        Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                       nm, _tr_g(v,10), _tr_g(s,9),
                       z, p, _tr_g(lo,11), _tr_g(hi,10))
    end
    println("-"^13, "+", "-"^64)

    # Derived
    for (nm, v, s, lo, hi) in (
        ("rho",    ρ̂, se_ρ, ρ_ci[1], ρ_ci[2]),
        ("sigma",  σ̂, se_σ, σ_ci[1], σ_ci[2]),
        ("lambda", λ̂, se_λ, λ_ci[1], λ_ci[2]))
        Printf.@printf("%12s | %s  %s  %7s  %6s  %s  %s\n",
                       nm, _tr_g(v,10), _tr_g(s,9),
                       "", "", _tr_g(lo,11), _tr_g(hi,10))
    end
    println("-"^78)
    Printf.@printf("LR test of indep. eqns. (rho = 0):   chi2(1) = %8.2f   Prob > chi2 = %.4f\n",
                   LR, p_LR)
    return (; β = β̂, γ = γ̂, ρ = ρ̂, σ = σ̂, λ = λ̂,
              logL, Wald, df_wald, p_wald, LR, p_LR,
              names_out, names_treat, V, se)
end
