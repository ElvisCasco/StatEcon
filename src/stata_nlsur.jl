import Optim

"""
    stata_nlsur(df, depvars, regs; vce=:robust, level=0.95,
                              quiet=false)
        -> NamedTuple

Stata `nlsur (y1 = normal(<linear>)) (y2 = normal(<linear>)),
vce(robust) nolog` for the two-equation case in which both equations
share the same `regs` list (textbook §15.10.2). The model is

    E[y_jᵢ | x_i] = Φ(α_j' · [1; x_i])    for j = 1, 2.

Two-step FGLS:

1. Minimise SSR = Σᵢ [r_1ᵢ² + r_2ᵢ²] (unweighted) to get α̃.
2. Form Σ̂ = (1/N) Σᵢ rᵢ rᵢ' from the residuals.
3. Minimise Σᵢ rᵢ' Σ̂⁻¹ rᵢ to get α̂ (FGLS).

`vce = :robust` (default): sandwich SE built from the central-FD
Jacobian of each equation's residual and the per-obs cross-product
of residuals.

Returns `(; α1, α2, Σhat, se1, se2, V, ll, n, regs, depvars, nparam)`.
"""
function stata_nlsur(df::DataFrames.AbstractDataFrame,
                                  depvars::AbstractVector{Symbol},
                                  regs::AbstractVector{Symbol};
                                  vce::Symbol = :robust,
                                  level::Float64 = 0.95,
                                  quiet::Bool = false)
    length(depvars) == 2 ||
        error("nlsur_two_eq needs exactly 2 dependent variables")
    cols = unique(vcat(collect(depvars), collect(regs)))
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c15_raw.(col))
        end
    end

    y1 = Float64.(_c15_raw.(dfc[!, depvars[1]]))
    y2 = Float64.(_c15_raw.(dfc[!, depvars[2]]))
    X  = hcat([Float64.(_c15_raw.(dfc[!, r])) for r in regs]..., ones(length(y1)))
    N, k = size(X)    # k = length(regs) + 1 (intercept is the LAST column,
                       # so the standard ordering is [regs...; _cons], matching
                       # Stata's `{a1}*age + {a2}*linc + {a3}*ndisease + {a4}`)
    nparam = 2 * k
    coefnames = [string.(regs)..., "_cons"]
    Φ_(x) = 0.5 * (1 + _c15_erf(x / sqrt(2.0)))

    function residuals(α)
        α1 = view(α, 1:k); α2 = view(α, (k + 1):(2k))
        r1 = y1 .- Φ_.(X * α1)
        r2 = y2 .- Φ_.(X * α2)
        return r1, r2
    end

    function ssr(α)
        r1, r2 = residuals(α)
        return sum(r1.^2) + sum(r2.^2)
    end

    function gls_obj(α, Σinv)
        r1, r2 = residuals(α)
        return Σinv[1,1] * sum(r1.^2) + 2 * Σinv[1,2] * sum(r1 .* r2) +
               Σinv[2,2] * sum(r2.^2)
    end

    # Step 1 — unweighted NLS.
    α0 = zeros(nparam)
    res1 = _c15_optimize(ssr, α0, Optim.LBFGS(),
                          Optim.Options(g_tol = 1e-8, iterations = 4000);
                          autodiff = :forward)
    α̃ = Optim.minimizer(res1)
    r1, r2 = residuals(α̃)
    Σhat = [sum(r1.^2)/N      sum(r1.*r2)/N;
            sum(r1.*r2)/N     sum(r2.^2)/N]
    Σinv = LinearAlgebra.inv(Σhat)

    # Step 2 — FGLS.
    res2 = _c15_optimize(α -> gls_obj(α, Σinv), α̃, Optim.LBFGS(),
                          Optim.Options(g_tol = 1e-8, iterations = 4000);
                          autodiff = :forward)
    α̂ = Optim.minimizer(res2)
    r1, r2 = residuals(α̂)

    # Robust sandwich SE.
    # The score for each obs: s_i = G_i' Σ̂⁻¹ r_i, where
    #   G_i = -[∂Φ(X_iα1)/∂α1 0; 0 ∂Φ(X_iα2)/∂α2]
    # G_i is 2 × 2k. Bread = (Σᵢ G_iᵀ Σ̂⁻¹ G_i)⁻¹.
    # Meat = Σᵢ G_iᵀ Σ̂⁻¹ r_i r_iᵀ Σ̂⁻¹ G_i.
    # V = bread · meat · bread'.
    φ_(x) = exp(-x^2 / 2) / sqrt(2 * Base.pi)
    bread_acc = zeros(nparam, nparam)
    meat_acc  = zeros(nparam, nparam)
    for i in 1:N
        xi = X[i, :]
        η1 = LinearAlgebra.dot(xi, α̂[1:k])
        η2 = LinearAlgebra.dot(xi, α̂[(k+1):2k])
        # G_i has shape 2 × 2k:
        #   row 1: [-φ(η1)·xi'   0]
        #   row 2: [0            -φ(η2)·xi']
        G = zeros(2, nparam)
        G[1, 1:k]      = -φ_(η1) .* xi
        G[2, (k+1):2k] = -φ_(η2) .* xi
        ri = [r1[i]; r2[i]]
        bread_acc .+= G' * Σinv * G
        meat_acc  .+= G' * Σinv * ri * ri' * Σinv * G
    end
    bread = LinearAlgebra.inv(LinearAlgebra.Symmetric(bread_acc))
    V = bread * meat_acc * bread'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    se1 = se[1:k]; se2 = se[(k+1):2k]

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
                       "Nonlinear SUR (two-eq, Φ-link)",
                       "Number of obs", commafmt(N))
        Printf.@printf("Σ̂ residual covariance:\n  [%.6f  %.6f;\n   %.6f  %.6f]\n",
                       Σhat[1,1], Σhat[1,2], Σhat[2,1], Σhat[2,2])
        println()
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Robust SE      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)
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
        Printf.@printf("%-12s |\n", string(depvars[1]))
        for r in 1:k
            _print(coefnames[r], α̂[r], se1[r])
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |\n", string(depvars[2]))
        for r in 1:k
            _print(coefnames[r], α̂[k + r], se2[r])
        end
        println("-"^78)
    end

    return (; α1 = α̂[1:k], α2 = α̂[(k+1):2k], Σhat,
              se1, se2, V, n = N,
              regs = collect(regs), depvars = collect(depvars), nparam)
end
