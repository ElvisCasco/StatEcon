# (deps provided by the StatEcon module)
import Optim
import ForwardDiff

"""
    stata_heckman(df, outcome, selection; vce=:oim, level=0.95,
                  quiet=false) -> NamedTuple

Stata `heckman <outcome eqn>, select(<selection eqn>) nolog` — full-information
maximum likelihood for the Heckman sample-selection model. `outcome` and
`selection` are `StatsModels` formulas, e.g.

    stata_heckman(df, @formula(lny ~ age + female + educ),
                      @formula(dy  ~ age + female + educ + income))

The model is

    y₁* = Xβ + ε          (outcome, observed only when the binary selector = 1)
    y₂* = Zγ + u          (latent selection index; `selection.lhs` is 1{y₂*>0})
    (ε, u) ~ N(0, [σ²  ρσ;  ρσ  1])

with per-observation log-likelihood

  * selected  : `log[φ((y₁ − Xβ)/σ)/σ] + log Φ((Zγ + (y₁ − Xβ)ρ/σ)/√(1−ρ²))`
  * unselected: `log Φ(−Zγ)`.

σ = exp(lnσ) and ρ = tanh(athrho) are optimised unconstrained (L-BFGS +
ForwardDiff gradient); OIM SEs come from a finite-difference Hessian with a
delta-method transform back to (σ, ρ). The intercept is placed last and
named `_cons`.

Returns `(; β, se, V, coefnames, rho, sigma, lambda, ll, n, …)` where `β`/`se`
are the OUTCOME-equation coefficients and their standard errors (constant
last), `rho` = ρ̂, `sigma` = σ̂, `lambda` = ρ̂σ̂, `ll` the log-likelihood and `n`
the sample size. The NamedTuple also carries the selection block `γ`/`se_γ`,
the regressor lists `outcome_regs`/`selection_regs`, and `σ`/`ρ` so
[`stata_predict_heckman`](@ref) can reuse the fit.
"""
function stata_heckman(df::DataFrames.AbstractDataFrame,
                       outcome::StatsModels.FormulaTerm,
                       selection::StatsModels.FormulaTerm;
                       vce::Symbol = :oim,
                       level::Float64 = 0.95,
                       quiet::Bool = false)
    outcome_dep    = Symbol(outcome.lhs)
    selection_dep  = Symbol(selection.lhs)
    outcome_regs   = collect(StatsModels.termvars(outcome.rhs))
    selection_regs = collect(StatsModels.termvars(selection.rhs))

    cols = unique(vcat([outcome_dep, selection_dep],
                       outcome_regs, selection_regs))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c16_rawval.(col))
        end
    end
    dy = Int.(_c16_rawval.(dfc[!, selection_dep]))
    y  = Float64.(_c16_rawval.(dfc[!, outcome_dep]))
    Xm = hcat([Float64.(_c16_rawval.(dfc[!, r])) for r in outcome_regs]...,
              ones(length(y)))
    Zm = hcat([Float64.(_c16_rawval.(dfc[!, r])) for r in selection_regs]...,
              ones(length(y)))
    N   = length(y); k_x = size(Xm, 2); k_z = size(Zm, 2)
    nparam = k_x + k_z + 2

    function negll(θ)
        β   = view(θ, 1:k_x)
        γ   = view(θ, (k_x + 1):(k_x + k_z))
        lnσ = θ[k_x + k_z + 1]
        ath = θ[k_x + k_z + 2]
        σ   = exp(lnσ)
        ρ   = tanh(ath)
        s   = sqrt(max(1 - ρ^2, eps(Float64)))
        Xβ = Xm * β; Zγ = Zm * γ
        ll = zero(eltype(θ))
        for i in 1:N
            if dy[i] == 1
                u = (y[i] - Xβ[i]) / σ
                a = (Zγ[i] + ρ * u) / s
                ll += -log(σ) - 0.5 * log(2 * Base.pi) - 0.5 * u^2 +
                      Distributions.logcdf(Distributions.Normal(), a)
            else
                ll += Distributions.logcdf(Distributions.Normal(), -Zγ[i])
            end
        end
        return -ll
    end

    # Warm start: OLS on selected obs for β + lnσ; γ = 0.
    pos = dy .== 1
    Xp  = Xm[pos, :]; yp = y[pos]
    βw  = Xp \ yp
    σw  = Statistics.std(yp .- Xp * βw)
    γw  = zeros(k_z)
    θ0  = vcat(βw, γw, log(σw), atanh(0.3))
    g!(G, x) = ForwardDiff.gradient!(G, negll, x)
    res = Optim.optimize(negll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 8000))
    θ̂  = Optim.minimizer(res)
    ll = -negll(θ̂)

    V_raw = LinearAlgebra.inv(LinearAlgebra.Symmetric(_c16_fd_hessian(negll, θ̂)))
    σ̂ = exp(θ̂[k_x + k_z + 1])
    ρ̂ = tanh(θ̂[k_x + k_z + 2])
    Jmat = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    Jmat[k_x + k_z + 1, k_x + k_z + 1] = σ̂          # dσ/dlnσ
    Jmat[k_x + k_z + 2, k_x + k_z + 2] = 1 - ρ̂^2    # dρ/dathrho
    V  = Jmat * V_raw * Jmat'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    β̂  = θ̂[1:k_x]
    γ̂  = θ̂[(k_x + 1):(k_x + k_z)]
    se_β = se[1:k_x]
    se_γ = se[(k_x + 1):(k_x + k_z)]
    se_σ = se[k_x + k_z + 1]
    se_ρ = se[k_x + k_z + 2]
    out_names = [string.(outcome_regs)..., "_cons"]
    sel_names = [string.(selection_regs)..., "_cons"]

    if !quiet
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
        crit = Distributions.quantile(Distributions.Normal(),
                                      1 - (1 - level) / 2)
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Heckman selection model",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6s\n", "",
                       "  Selected", commafmt(sum(dy)))
        Printf.@printf("%56s%-13s = %6s\n", "",
                       "  Nonselected", commafmt(N - sum(dy)))
        Printf.@printf("Log likelihood = %.4f\n\n", ll)

        function _print(label, b, s)
            z = b / s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit*s; hi = b + crit*s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", string(outcome_dep))
        for r in vcat(1:(k_x - 1), [k_x])
            _print(out_names[r], β̂[r], se_β[r])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", string(selection_dep))
        for r in vcat(1:(k_z - 1), [k_z])
            _print(sel_names[r], γ̂[r], se_γ[r])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", "/athrho")
        _print("athrho", θ̂[end], sqrt(V_raw[end, end]))
        Printf.@printf("%-12s |\n", "/lnsigma")
        _print("lnsigma", θ̂[end - 1], sqrt(V_raw[end - 1, end - 1]))
        Printf.@printf("%-12s |\n", "rho")
        _print("rho", ρ̂, se_ρ)
        Printf.@printf("%-12s |\n", "sigma")
        _print("sigma", σ̂, se_σ)
        Printf.@printf("%-12s |\n", "lambda")
        _print("lambda", ρ̂ * σ̂, NaN)
        println("-"^78)
        χ2_ind = (θ̂[end] / sqrt(V_raw[end, end]))^2
        p_ind  = 1 - Distributions.cdf(Distributions.Chisq(1), χ2_ind)
        Printf.@printf("LR test of ρ = 0: chi2(1) = %.2f   Prob > chi2 = %.4f\n",
                       χ2_ind, p_ind)
    end

    return (; β = β̂, se = se_β, V = V, coefnames = out_names,
              rho = ρ̂, sigma = σ̂, lambda = ρ̂ * σ̂, ll = ll, n = N,
              γ = γ̂, σ = σ̂, ρ = ρ̂,
              se_β = se_β, se_γ = se_γ, se_σ = se_σ, se_ρ = se_ρ,
              selection_coefnames = sel_names,
              outcome_regs = collect(outcome_regs),
              selection_regs = collect(selection_regs),
              depvars = [outcome_dep, selection_dep],
              n_sel_1 = sum(dy), n_sel_0 = N - sum(dy), nparam = nparam)
end
