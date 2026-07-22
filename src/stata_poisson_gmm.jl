# ============================================================================
# stata_poisson_gmm.jl — nonlinear-IV (GMM) Poisson (Cameron & Trivedi ch17)
#
# The GMM objective is minimised by Optim.Newton with an analytic gradient and
# Gauss-Newton Hessian (the Mata `pgmm` d2 evaluator), so no autodiff is used.
# `_c17_rawval` (defined in stata_ztnb.jl) unwraps labeled columns.
# ============================================================================

import Optim

"""
    stata_poisson_gmm(df, depvar, xvars, zvars; twostep=false, level=0.95,
                      quiet=false) -> NamedTuple

Nonlinear-IV (GMM) Poisson — the Julia counterpart of the textbook's Mata
`pgmm` program. Moment conditions E[Z'(y − exp(Xβ))] = 0 with
   Q(β) = h'W h,   h = Z'(y − μ),   μ = exp(Xβ).
`xvars` are the count regressors (including the endogenous one), `zvars`
the instruments (excluded instruments + exogenous regressors); a constant
is appended to both X and Z. One-step uses W = (Z'Z)⁻¹; `twostep=true`
re-weights by the optimal Ŝ⁻¹ from first-step residuals.

Robust (GMM-sandwich) vcov, matching the Mata code:
   G = −(μ ⊙ Z)'X,  Ŝ = ((y−μ)⊙Z)'((y−μ)⊙Z)·n/(n−k),
   V = (G'WG)⁻¹ G'W Ŝ W G (G'WG)⁻¹.

Prints a Stata-style coefficient table. Returns `(; β, V, se, coefnames,
X, Z, y, n, k, W, df, twostep, Jstat, J_df, J_p)`.
"""
function stata_poisson_gmm(df, depvar::Symbol,
                           xvars::AbstractVector{Symbol},
                           zvars::AbstractVector{Symbol};
                           twostep::Bool = false, level::Float64 = 0.95,
                           quiet::Bool = false)
    cols = unique(vcat([depvar], collect(xvars), collect(zvars)))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c17_rawval.(col))
        end
    end
    y = Float64.(_c17_rawval.(dfc[!, depvar])); n = length(y)
    X = hcat([Float64.(_c17_rawval.(dfc[!, v])) for v in xvars]..., ones(n))
    Z = hcat([Float64.(_c17_rawval.(dfc[!, v])) for v in zvars]..., ones(n))
    k = size(X, 2)
    xnames = vcat(string.(xvars), "_cons")

    Wmat = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Z'Z)))
    # Objective Q(β)=h'Wh with analytic gradient ∇Q = 2·G'W·h and the
    # Gauss-Newton Hessian 2·G'WG, where h = Z'(y−μ), μ = exp(Xβ),
    # G = ∂h/∂β = −(μ⊙Z)'X. This is exactly the Mata `pgmm` d2
    # evaluator; Newton's method converges from β = 0 (plain LBFGS
    # stalls on this surface), and it sidesteps the autodiff API.
    Qf(W) = β -> (h = Z' * (y .- exp.(X*β)); h' * W * h)
    Qg!(W) = (g, β) -> begin
        μ = exp.(X*β); h = Z' * (y .- μ); G = -(μ .* Z)' * X
        g .= 2 .* (G' * (W * h)); g
    end
    Qh!(W) = (H, β) -> begin
        μ = exp.(X*β); G = -(μ .* Z)' * X
        H .= 2 .* (G' * W * G); H
    end
    _fit(W, β0) = Optim.minimizer(Optim.optimize(Qf(W), Qg!(W), Qh!(W), β0,
                       Optim.Newton(), Optim.Options(iterations = 1000,
                                                     g_tol = 1e-10)))
    β = _fit(Wmat, zeros(k))
    if twostep
        e1   = y .- exp.(X*β)
        Wmat = LinearAlgebra.inv(((e1 .* Z)' * (e1 .* Z)) / n)
        β    = _fit(Wmat, β)
    end

    μ    = exp.(X*β)
    G    = -(μ .* Z)' * X
    Shat = ((y .- μ) .* Z)' * ((y .- μ) .* Z) * (n / (n - k))
    GWG  = G' * Wmat * G
    GWGi = LinearAlgebra.inv(GWG)
    V    = GWGi * G' * Wmat * Shat * Wmat * G * GWGi
    se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z    = β ./ se
    p    = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)

    if !quiet
        function g9(x; w::Int=10, sig::Int=7)
            (ismissing(x)||!isfinite(x)) && return lpad(".", w)
            su=sig; s=Printf.@sprintf("%.*g", su, x); cap=(0<abs(x)<1 && x<0) ? 10 : 9
            while length(s)>cap && su>1; su-=1; s=Printf.@sprintf("%.*g",su,x); end
            0<abs(x)<1 && (s=replace(s, r"^(-?)0\."=>s"\1.")); lpad(s,w)
        end
        commafmt(num::Integer)=begin s=string(abs(num)); ps=String[]; i=length(s)
            while i>=1; push!(ps,s[max(1,i-2):i]); i-=3; end; (num<0 ? "-" : "")*join(reverse(ps),",") end
        println()
        Printf.@printf("%-52s%-13s = %8s\n",
                       twostep ? "Poisson GMM (two-step)" : "Poisson GMM (one-step)",
                       "Number of obs", commafmt(n))
        println("-"^78)
        println("             |               Robust")
        Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100*level)
        println("-"^13, "+", "-"^64)
        ord = vcat(1:(k-1), [k])    # slopes then _cons
        for i in ord
            lo=β[i]-crit*se[i]; hi=β[i]+crit*se[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           xnames[i], g9(β[i];w=10), g9(se[i];w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", p[i]),
                           g9(lo;w=9), g9(hi;w=10))
        end
        println("-"^78)
    end

    # Hansen's J statistic for the test of overidentifying restrictions
    # (`estat overid`). With h = Z'(y−μ̂) = N·ḡ and the two-step optimal
    # weight W = Ŝ⁻¹, J = N·ḡ'Ŝ⁻¹ḡ = (h'Wh)/N ~ χ²(#instruments −
    # #params). Only valid after the *two-step* fit (W = Ŝ⁻¹); for
    # one-step it is reported but not the efficient-GMM J.
    h_final = Z' * (y .- exp.(X*β))
    Jstat = (h_final' * Wmat * h_final) / n
    J_df  = size(Z, 2) - size(X, 2)
    J_p   = J_df > 0 ? 1 - Distributions.cdf(Distributions.Chisq(J_df), Jstat) : NaN

    return (; β, V, se, coefnames = xnames, X, Z, y, n, k, W = Wmat, df = dfc,
              twostep, Jstat, J_df, J_p)
end
