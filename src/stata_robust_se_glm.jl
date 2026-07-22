# ============================================================================
# stata_robust_se_glm.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_robust_se_glm(model)

Manual robust (HC1) sandwich standard errors after a Poisson `GLM.glm` fit
with a log link:

    V = H⁻¹ · (Σ xᵢ xᵢ' (yᵢ − μ̂ᵢ)²) · H⁻¹ · n/(n−k),   H = X' diag(μ̂) X

Returns the vector of robust standard errors (√diag V) in GLM coefficient
order. `stata_poisson(...; vce = :robust)` applies the same correction and is
the preferred entry point; this helper is exposed for ad-hoc use on a raw
`GLM` model object.
"""
function stata_robust_se_glm(model)
    Xm   = GLM.modelmatrix(model)
    yv   = Float64.(GLM.response(model))
    μ    = GLM.predict(model)
    n, k = size(Xm)
    H    = Xm' * LinearAlgebra.Diagonal(μ) * Xm
    meat = Xm' * LinearAlgebra.Diagonal((yv .- μ).^2) * Xm
    Hinv = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H)))
    V    = Hinv * meat * Hinv * (n / (n - k))
    return sqrt.(max.(LinearAlgebra.diag(V), 0.0))
end
