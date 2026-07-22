# ============================================================================
# stata_nl_exp.jl — Stata `nl (y = exp(xb))` NLS (Cameron & Trivedi ch17)
#
# Nonlinear least squares with an exponential mean. The sum-of-squares is
# smooth, so the source's `autodiff=:forward` is realised here with an explicit
# ForwardDiff gradient + the 5-arg `Optim.optimize(f, g!, x0, method, opts)`.
# `_c17_rawval` (defined in stata_ztnb.jl) unwraps labeled columns.
# ============================================================================

import Optim
import ForwardDiff

"""
    stata_nl_exp(df, depvar, regs; vce=:robust, level=0.95, quiet=false,
                 β0=nothing) -> NamedTuple

Stata's `nl (y = exp({xb: <regs> _cons})), vce(robust|oim)` — nonlinear
least squares with exponential mean function:

    y_i = exp(x_i'β) + u_i,    minimise Σ(y_i − exp(x_i'β))²

For Poisson-like count data this is the "NLS-Poisson" estimator, identical
to Poisson MLE only when the conditional mean is correctly specified but
without exploiting the Poisson variance structure. SEs default to the
Huber–White sandwich (analogous to Stata's `vce(robust)`):

    V = (G'G)⁻¹ · G'·diag((y − μ̂)²)·G · (G'G)⁻¹ · n/(n − k)

with G_i = μ̂_i · x_i (Jacobian of the mean function).

Prints the Stata-style "Nonlinear regression" block (Number of obs, R²,
Root MSE) plus the coefficient table, with "/xb_<regname>" parameter
labels matching Stata's `{xb: …}` notation.

Returns `(; β, V, se, μ̂, residuals, ll, RSS, R2, RMSE, n, k, coefnames,
vce_label)`.
"""
function stata_nl_exp(df::DataFrames.AbstractDataFrame, depvar::Symbol,
                     regs::AbstractVector{Symbol};
                     vce::Symbol = :robust, level::Float64 = 0.95,
                     quiet::Bool = false,
                     β0::Union{Nothing, AbstractVector} = nothing)
    cols = vcat([depvar], collect(regs))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c17_rawval.(col))
        end
    end
    y  = Float64.(_c17_rawval.(dfc[!, depvar]))
    Xm = hcat([Float64.(_c17_rawval.(dfc[!, r])) for r in regs]..., ones(length(y)))
    n  = length(y); k = size(Xm, 2)
    coefnames = vcat(string.(regs), "_cons")

    obj(β) = sum((y .- exp.(Xm * β)).^2)
    β_start = β0 === nothing ?
              vcat(zeros(k - 1), log(max(Statistics.mean(y), eps()))) :
              collect(β0)
    g! = (g, x) -> ForwardDiff.gradient!(g, obj, x)
    res = Optim.optimize(obj, g!, β_start, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 500))
    β̂  = Optim.minimizer(res)
    μ̂  = exp.(Xm * β̂)
    e  = y .- μ̂
    RSS = sum(e.^2)
    σ̂² = RSS / (n - k)
    RMSE = sqrt(σ̂²)
    TSS = sum((y .- Statistics.mean(y)).^2)
    R2  = 1 - RSS / TSS

    # Jacobian rows: ∂μ_i/∂β = μ_i · x_i
    G = μ̂ .* Xm
    GtG = G' * G
    GtG_inv = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(GtG)))
    if vce == :robust
        meat = G' * LinearAlgebra.Diagonal(e.^2) * G
        V = GtG_inv * meat * GtG_inv * (n / (n - k))
        vce_label = "Robust"
    else
        V = GtG_inv * σ̂²
        vce_label = "OIM"
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β̂ ./ se
    p  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ll  = -n/2 * log(2 * Base.pi * σ̂²) - n/2

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

    if !quiet
        println()
        Printf.@printf("%-50s%-13s = %10s\n",
                       "Nonlinear regression",
                       "Number of obs", commafmt(n))
        Printf.@printf("%50s%-13s = %10.4f\n", "", "R-squared", R2)
        Printf.@printf("%50s%-13s = %10.4f\n", "", "Root MSE", RMSE)
        Printf.@printf("%50s%-13s = %10.4f\n", "", "Res. dev.",
                       n * log(RSS / n))
        println()
        println("-"^78)
        if vce == :robust
            println("             |               Robust")
        end
        se_label = vce == :robust ? "std. err." : "Std. err."
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), se_label, 100 * level)
        println("-"^13, "+", "-"^64)
        for i in 1:k
            lab = coefnames[i] == "_cons" ? "/xb_one" : "/xb_$(coefnames[i])"
            lo = β̂[i] - crit * se[i]; hi = β̂[i] + crit * se[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           lab, g9(β̂[i]; w=10), g9(se[i]; w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", p[i]),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^78)
    end

    return (; β = β̂, V, se, μ̂, residuals = e,
              ll, RSS, R2, RMSE, n, k, coefnames, vce_label)
end
