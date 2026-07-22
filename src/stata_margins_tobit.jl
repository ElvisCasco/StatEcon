# (deps provided by the StatEcon module)

"""
    stata_margins_tobit(res; predict=:e_cond, at=:mean, lower=nothing,
                        upper=nothing, level=0.95, quiet=false) -> NamedTuple

Stata `margins, dydx(*) predict(...)` after `tobit`. `res` is the NamedTuple
returned by [`stata_tobit`](@ref). The `predict` keyword selects the quantity
whose marginal effect is computed:

* `:xb`       — linear index xβ
* `:e_cond`   — E[y | L < y < U, x]  (Stata `predict(e(L, U))`)
* `:e_uncond` — censored expectation E[y* | x], y* = max(L, min(y, U))
                (Stata `predict(ystar(L, U))`)
* `:pr`       — P(L < y < U | x)  (Stata `predict(pr(L, U))`)

`at = :mean` (Stata's `atmean`, MEM) evaluates at the column means of `res.X`;
`at = :asobserved` (AME) averages the per-observation marginal effect. The
censoring bounds default to `res.ll_bound` / `res.ul_bound`; override with the
`lower` / `upper` keywords. Standard errors are delta-method: a central
finite-difference Jacobian of the mean marginal effect in (β, lnσ) against
`res.V`.

Returns `(; rows, predict, at, lower, upper, n)`.
"""
function stata_margins_tobit(res; predict::Symbol = :e_cond,
                             at = :mean,
                             lower::Union{Nothing,Real} = nothing,
                             upper::Union{Nothing,Real} = nothing,
                             level::Float64 = 0.95,
                             quiet::Bool = false)
    L = lower === nothing ?
        (res.ll_bound === nothing ? -Inf : Float64(res.ll_bound)) :
        Float64(lower)
    U = upper === nothing ?
        (res.ul_bound === nothing ?  Inf : Float64(res.ul_bound)) :
        Float64(upper)

    X = res.X; k = size(X, 2)
    Xeval = at === :mean       ? reshape(vec(Statistics.mean(X; dims = 1)),
                                         1, k) :
            at === :asobserved ? X :
            reshape(convert(Vector{Float64}, at), 1, k)

    Φ_(z) = _c16_Phi(z)
    φ_(z) = _c16_phi(z)

    # Analytical per-row marginal effect ∂(predicted)/∂x_c for each
    # `predict` type. Closed forms let the delta-method operate cleanly on θ
    # via a single outer finite difference.
    function me_at(θ, c)
        β = view(θ, 1:k); σ = exp(θ[end])
        Nrows = size(Xeval, 1); acc = zero(eltype(θ))
        for i in 1:Nrows
            xi = view(Xeval, i, :)
            η  = LinearAlgebra.dot(xi, β)
            βc = β[c]
            row_me = if predict === :xb
                βc
            elseif predict === :pr
                a_phi = isfinite(L) ? φ_((L - η)/σ) : zero(eltype(θ))
                b_phi = isfinite(U) ? φ_((U - η)/σ) : zero(eltype(θ))
                βc / σ * (a_phi - b_phi)
            elseif predict === :e_uncond
                Φa = isfinite(L) ? Φ_((L - η)/σ) : zero(eltype(θ))
                Φb = isfinite(U) ? Φ_((U - η)/σ) :  one(eltype(θ))
                βc * (Φb - Φa)
            elseif predict === :e_cond
                a = isfinite(L) ? (L - η)/σ : -convert(eltype(θ), Inf)
                b = isfinite(U) ? (U - η)/σ :  convert(eltype(θ), Inf)
                Φa = isfinite(a) ? Φ_(a) : zero(eltype(θ))
                Φb = isfinite(b) ? Φ_(b) :  one(eltype(θ))
                φa = isfinite(a) ? φ_(a) : zero(eltype(θ))
                φb = isfinite(b) ? φ_(b) : zero(eltype(θ))
                denom = max(Φb - Φa, eps(Float64))
                λ_a = φa / denom
                λ_b = φb / denom
                a_term = isfinite(a) ? a * λ_a : zero(eltype(θ))
                b_term = isfinite(b) ? b * λ_b : zero(eltype(θ))
                βc * (1 - (λ_a - λ_b)^2 + a_term - b_term)
            else
                error("unknown predict type: $predict")
            end
            acc += row_me
        end
        return acc / Nrows
    end

    θ̂ = vcat(res.β, log(res.σ))
    # `res.V` is in (β, σ) units; convert back to (β, lnσ) for the delta
    # method: V_raw = J⁻¹ V (J⁻¹)' with J = diag(1, …, 1, σ).
    Jinv = Matrix{Float64}(LinearAlgebra.I, k + 1, k + 1)
    Jinv[end, end] = 1 / res.σ
    V_raw = Jinv * res.V * Jinv'

    crit = Distributions.quantile(Distributions.Normal(),
                                  1 - (1 - level) / 2)
    h_step = sqrt(eps(Float64))
    rows = NamedTuple[]
    for c in 1:(k - 1)              # slope coefficients (exclude constant)
        m_pt = me_at(θ̂, c)
        g = zeros(length(θ̂))
        for i in eachindex(θ̂)
            step = max(abs(θ̂[i]) * h_step, h_step)
            tp = copy(θ̂); tp[i] += step
            tm = copy(θ̂); tm[i] -= step
            g[i] = (me_at(tp, c) - me_at(tm, c)) / (2 * step)
        end
        se = sqrt(max(g' * V_raw * g, 0.0))
        z  = m_pt / se
        p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        push!(rows, (; reg = Symbol(res.coefnames[c]),
                      dydx = m_pt, se, z, p,
                      ci_lo = m_pt - crit * se, ci_hi = m_pt + crit * se))
    end

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
        N = size(X, 1)
        println()
        title = at === :asobserved ? "Average marginal effects" :
                                     "Conditional marginal effects"
        Printf.@printf("%-56s%-13s = %6s\n", title,
                       "Number of obs", commafmt(N))
        println("Model VCE: OIM")
        expr_str = predict === :xb       ? "linear prediction xβ" :
                   predict === :e_cond   ? "E[y | $(L) < y < $(U)]" :
                   predict === :e_uncond ? "E[y* | x]  (censored at $(L), $(U))" :
                   predict === :pr       ? "P($(L) < y < $(U))" : "?"
        println("Expression: ", expr_str)
        println()
        println("-"^78)
        println("             |            Delta-method")
        Printf.@printf("%12s |      dy/dx   Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)
        for r in rows
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           string(r.reg),
                           g9(r.dydx; w = 10), g9(r.se; w = 9),
                           Printf.@sprintf("%7.2f", r.z),
                           Printf.@sprintf("%.3f", r.p),
                           g9(r.ci_lo; w = 9), g9(r.ci_hi; w = 10))
        end
        println("-"^78)
    end

    return (; rows, predict, at = Xeval, lower = L, upper = U, n = size(X, 1))
end
