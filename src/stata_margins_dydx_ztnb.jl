# ============================================================================
# stata_margins_dydx_ztnb.jl — `margins, dydx(*)` after `ztnb`  (C&T ch17)
# ============================================================================

"""
    stata_margins_dydx_ztnb(fit; specs, atmean=true, level=0.95,
                            depvar="y", vce_label="Robust",
                            noatlegend=false) -> NamedTuple

Stata's `margins, dydx(*)` after `ztnb` — marginal effects on the
zero-truncated NB2 conditional mean

    m(x) = μ / (1 − P(Y=0)),  μ = exp(x'β),  P(Y=0) = (1 + α·μ)^{−1/α}

`fit` is a `stata_ztnb` result (supplies β in GLM order — intercept
first — plus `lnα`, the full (k+1)×(k+1) vcov `V`, and the design
matrix `X`). `specs` is a vector of `(name, kind, col_idx)` with
`col_idx` into `fit.X` (GLM order: column 1 = intercept):
  - `:continuous` → dy/dx_j = β_j · ∂m/∂η at the evaluation point
  - `:factor`     → discrete change m(·|x_j=1) − m(·|x_j=0)

`atmean=true` evaluates at the column means of `X` (Stata `atmean`);
`atmean=false` averages the effect over the sample (AME). Delta-method
SEs use a finite-difference gradient of the margin w.r.t. θ = (β, lnα)
against the full vcov — no fragile analytic Jacobian through the
truncation term. Prints the Stata `Conditional/Average marginal
effects` block; factor rows are labelled `1.<name>`.

Returns `(; rows)` with `(name, me, se, z, p, ci_lo, ci_hi)` each.
"""
function stata_margins_dydx_ztnb(fit;
                                 specs::AbstractVector,
                                 atmean::Bool = true,
                                 level::Float64 = 0.95,
                                 depvar::AbstractString = "y",
                                 vce_label::AbstractString = "Robust",
                                 noatlegend::Bool = false)
    X = fit.X; n, k = size(X)
    θ̂ = vcat(fit.β, fit.lnα)           # (β, lnα), length k+1
    V = fit.V

    # Truncated NB2 mean for one row given parameter vector θ.
    _m(xrow, θ) = begin
        β = view(θ, 1:k); α = exp(θ[k + 1]); iα = 1 / α
        μ = exp(LinearAlgebra.dot(xrow, β))
        p0 = (1 + α * μ)^(-iα)
        μ / (1 - p0)
    end

    # dy/dx for variable `idx` at parameter θ, evaluated at point/sample.
    function _effect(kind, idx, θ)
        if atmean
            c̄ = vec(Statistics.mean(X, dims = 1))
            if kind == :factor
                c1 = copy(c̄); c1[idx] = 1.0
                c0 = copy(c̄); c0[idx] = 0.0
                return _m(c1, θ) - _m(c0, θ)
            else
                h = 1e-6 * max(abs(c̄[idx]), 1.0)
                cp = copy(c̄); cp[idx] += h
                cm = copy(c̄); cm[idx] -= h
                return (_m(cp, θ) - _m(cm, θ)) / (2h)
            end
        else
            acc = 0.0
            for i in 1:n
                xi = X[i, :]
                if kind == :factor
                    c1 = copy(xi); c1[idx] = 1.0
                    c0 = copy(xi); c0[idx] = 0.0
                    acc += _m(c1, θ) - _m(c0, θ)
                else
                    h = 1e-6 * max(abs(xi[idx]), 1.0)
                    cp = copy(xi); cp[idx] += h
                    cm = copy(xi); cm[idx] -= h
                    acc += (_m(cp, θ) - _m(cm, θ)) / (2h)
                end
            end
            return acc / n
        end
    end

    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    rows = NamedTuple[]
    for (name, kind, idx) in specs
        me = _effect(kind, idx, θ̂)
        # Delta-method gradient wrt θ by central finite differences.
        g = zeros(k + 1)
        for j in 1:(k + 1)
            hj = sqrt(eps(Float64)) * max(abs(θ̂[j]), 1.0)
            θp = copy(θ̂); θp[j] += hj
            θm = copy(θ̂); θm[j] -= hj
            g[j] = (_effect(kind, idx, θp) - _effect(kind, idx, θm)) / (2hj)
        end
        se = sqrt(max(LinearAlgebra.dot(g, V * g), 0.0))
        z  = me / se
        p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        push!(rows, (; name = kind == :factor ? "1.$name" : name,
                       me, se, z, p,
                       ci_lo = me - crit*se, ci_hi = me + crit*se))
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

    title = atmean ? "Conditional marginal effects" : "Average marginal effects"
    println()
    Printf.@printf("%-57s%-13s = %5s\n", title, "Number of obs", commafmt(n))
    Printf.@printf("Model VCE: %s\n\n", vce_label)
    println("Expression: Predicted mean, predict()")
    Printf.@printf("dy/dx wrt:  %s\n", join((r.name for r in rows), " "))
    if atmean && !noatlegend
        println("At: (means)")
    end
    println()
    println("-"^78)
    println("             |            Delta-method")
    Printf.@printf("             |      dy/dx   std. err.      z    P>|z|     [%g%% conf. interval]\n",
                   100*level)
    println("-"^13, "+", "-"^64)
    for r in rows
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       r.name, g9(r.me; w=10), g9(r.se; w=9),
                       Printf.@sprintf("%7.2f", r.z),
                       Printf.@sprintf("%.3f", r.p),
                       g9(r.ci_lo; w=9), g9(r.ci_hi; w=10))
    end
    println("-"^78)
    return (; rows)
end
