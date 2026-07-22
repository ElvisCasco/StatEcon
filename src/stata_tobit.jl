# (deps provided by the StatEcon module)
import Optim
import ForwardDiff

# ── shared ch16 helpers ────────────────────────────────────────────────
# Unwrap a ReadStatTables.LabeledValue -> its underlying numeric value.
_c16_rawval(x) = hasproperty(x, :value) ? x.value : x

# Standard-normal cdf / pdf via Distributions (ForwardDiff-differentiable
# through StatsFuns' DiffRules — no SpecialFunctions needed).
_c16_Phi(z) = Distributions.cdf(Distributions.Normal(), z)
_c16_phi(z) = Distributions.pdf(Distributions.Normal(), z)

# Central finite-difference Hessian of a scalar function `f` at `x`.
# Kept as an explicit FD routine (rather than ForwardDiff.hessian) because
# the log-likelihoods clamp their arguments with `max(., eps)`, which
# introduces kinks that break nested-Dual second differentiation. Shared by
# `stata_tobit` and `stata_heckman`.
function _c16_fd_hessian(f, x)
    nθ = length(x); H = zeros(nθ, nθ)
    h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
    for i in 1:nθ
        xpi = copy(x); xmi = copy(x); xpi[i] += h_[i]; xmi[i] -= h_[i]
        H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h_[i]^2
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

"""
    stata_tobit(df, depvar, indepvars; ll=nothing, ul=nothing,
                vce=:oim, level=0.95, quiet=false) -> NamedTuple

Stata `tobit <depvar> <indepvars>, ll(#) ul(#)` — maximum-likelihood
estimation of the censored (Tobit) regression model. `ll` sets the
left-censoring point (Stata's `ll(#)`; `nothing` means no left censoring),
`ul` the right-censoring point (`ul(#)`). Observations at or below `ll` are
treated as left-censored, at or above `ul` as right-censored, the rest as
uncensored:

  * uncensored : `log[φ((y − xβ)/σ)/σ]`
  * left-cens. : `log Φ((ll − xβ)/σ)`
  * right-cens.: `log Φ((xβ − ul)/σ)`

Fitted by L-BFGS with an explicit ForwardDiff gradient (`σ = exp(lnσ)` is
optimised unconstrained); the OIM VCE comes from a finite-difference Hessian
with a delta-method transform on `σ = exp(lnσ)`. The intercept is placed last
and named `_cons`. This is the canonical censored-regression estimator reused
by later chapters.

Returns `(; β, se, V, coefnames, sigma, ll, n, …)` where `β`/`se` are the
coefficient estimates and their OIM standard errors (constant last), `sigma`
is σ̂, `ll` is the maximised log-likelihood, and `n` the sample size. The
NamedTuple also carries `σ`, `se_σ`, the design matrix `X`, the regressor
list `regs`, the censoring bounds `ll_bound`/`ul_bound`, and the censored /
uncensored counts for use by `stata_margins_tobit` and the tobit predictors.
"""
function stata_tobit(df::DataFrames.AbstractDataFrame, depvar::Symbol,
                     indepvars::AbstractVector{Symbol};
                     ll::Union{Nothing,Real} = nothing,
                     ul::Union{Nothing,Real} = nothing,
                     vce::Symbol = :oim,
                     level::Float64 = 0.95,
                     quiet::Bool = false)
    cols = vcat([depvar], collect(indepvars))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c16_rawval.(col))
        end
    end
    y = Float64.(_c16_rawval.(dfc[!, depvar]))
    X = hcat([Float64.(_c16_rawval.(dfc[!, r])) for r in indepvars]...,
             ones(length(y)))
    N, k = size(X)
    nparam = k + 1                              # β (k) + lnσ
    coefnames = [string.(indepvars)..., "_cons"]

    function negll(θ)
        β = view(θ, 1:k); σ = exp(θ[end])
        η = X * β
        ll_acc = zero(eltype(θ))
        for i in 1:N
            yi = y[i]
            if ll !== nothing && yi <= ll
                ll_acc += Distributions.logcdf(Distributions.Normal(),
                                               (ll - η[i]) / σ)
            elseif ul !== nothing && yi >= ul
                ll_acc += Distributions.logcdf(Distributions.Normal(),
                                               (η[i] - ul) / σ)
            else
                z = (yi - η[i]) / σ
                ll_acc += -log(σ) - 0.5 * log(2 * Base.pi) - 0.5 * z^2
            end
        end
        return -ll_acc
    end

    # Warm start: OLS β on the uncensored part + lnσ from OLS residuals.
    function _ols_warm()
        mask = ll === nothing ? trues(N) : (y .> ll)
        if sum(mask) > k
            Xu = X[mask, :]; yu = y[mask]
            βu = Xu \ yu
            σu = Statistics.std(yu .- Xu * βu)
            return vcat(βu, log(max(σu, eps(Float64))))
        end
        return vcat(zeros(k), 0.0)
    end
    θ0 = _ols_warm()
    g!(G, x) = ForwardDiff.gradient!(G, negll, x)
    res = Optim.optimize(negll, g!, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 4000))
    θ̂  = Optim.minimizer(res)
    β̂  = θ̂[1:k]; σ̂ = exp(θ̂[end])
    llv = -negll(θ̂)

    V_raw = LinearAlgebra.inv(LinearAlgebra.Symmetric(_c16_fd_hessian(negll, θ̂)))
    # Delta-method on σ = exp(lnσ): dσ/dlnσ = σ.
    Jmat = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    Jmat[end, end] = σ̂
    V = Jmat * V_raw * Jmat'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se_β = se[1:k]; se_σ = se[end]

    n_left_cens  = ll === nothing ? 0 : sum(y .<= ll)
    n_right_cens = ul === nothing ? 0 : sum(y .>= ul)
    n_uncens     = N - n_left_cens - n_right_cens

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
        Printf.@printf("%-56s%-13s = %6s\n", "Tobit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6s\n", "",
                       "  Uncensored", commafmt(n_uncens))
        ll === nothing || Printf.@printf("%56s%-13s = %6s\n", "",
                       "  Left-cens.", commafmt(n_left_cens))
        ul === nothing || Printf.@printf("%56s%-13s = %6s\n", "",
                       "  Right-cens.", commafmt(n_right_cens))
        println(Printf.@sprintf("Log likelihood = %.4f\n", llv))

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      t    P>|t|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        for c in 1:k
            label = c == k ? "_cons" : string(indepvars[c])
            b = β̂[c]; s = se_β[c]; z = b/s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit*s; hi = b + crit*s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %s  %s\n", "/sigma",
                       g9(σ̂; w=10), g9(se_σ; w=9))
        println("-"^78)
    end

    return (; β = β̂, se = se_β, V = V, coefnames = coefnames,
              sigma = σ̂, ll = llv, n = N,
              σ = σ̂, se_β = se_β, se_σ = se_σ,
              X = X, regs = collect(indepvars),
              n_uncens = n_uncens, n_left_cens = n_left_cens,
              n_right_cens = n_right_cens, nparam = nparam,
              ll_bound = ll, ul_bound = ul,
              depvar_name = depvar, vce = vce)
end
