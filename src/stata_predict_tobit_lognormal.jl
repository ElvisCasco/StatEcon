# (deps provided by the StatEcon module)

"""
    stata_predict_tobit_lognormal(fit; σ=fit.σ, γ, X=fit.X,
                                  truncated=false) -> Vector

Lognormal-tobit prediction of `y = exp(ln_y)` from a [`stata_tobit`](@ref) fit
on `ln(y)` left-censored at `γ`. With xβ = X β̂:

  yhat_i   = exp(xβ_i + ½σ²)·(1 − Φ((γ − xβ_i − σ²)/σ))    (E[y | x])
  ytrunc_i = yhat_i / (1 − Φ((γ − xβ_i)/σ))                 (E[y | x, y>0])

`σ` defaults to the fitted σ̂ but can be overridden — used in the textbook's
"poor predictions are due to high σ" sensitivity check (σ = 1, σ = 2 vs σ̂).
Pass `truncated=true` for the positive-only conditional expectation.
"""
function stata_predict_tobit_lognormal(fit; σ::Real = fit.σ, γ::Real,
                                       X::AbstractMatrix = fit.X,
                                       truncated::Bool = false)
    Φ_(z) = _c16_Phi(z)
    xb = X * fit.β
    yhat = exp.(xb .+ 0.5 * σ^2) .*
           (1 .- Φ_.((γ .- xb .- σ^2) ./ σ))
    truncated || return yhat
    return yhat ./ (1 .- Φ_.((γ .- xb) ./ σ))
end
