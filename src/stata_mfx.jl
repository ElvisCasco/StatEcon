# ============================================================================
# stata_mfx.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_mfx(β, V, X; factors, continuous, kind=:dydx, model_name="poisson",
              expression="Predicted number of events (predict)",
              level=0.95, evaluate=:atmean, at=nothing)

Stata-style legacy `mfx, dydx varlist(...)` output — same dy/dx + delta-method
SE math as `stata_margins_dydx` but in the older `mfx` table layout (with a
trailing `X` column of regressor values at the evaluation point). `evaluate`
∈ (`:atmean`) or pass an explicit `at` vector. `kind` as in
`stata_margins_dydx`. Returns the row tuples.
"""
function stata_mfx(β::AbstractVector, V::AbstractMatrix, X::AbstractMatrix;
                   factors::AbstractVector, continuous::AbstractVector,
                   kind::Symbol=:dydx,
                   model_name::String="poisson",
                   expression::String="Predicted number of events (predict)",
                   level::Float64=0.95,
                   evaluate::Symbol=:atmean,
                   at::Union{Nothing,AbstractVector}=nothing)
    kind in (:dydx, :eyex, :eydx, :dyex) ||
        error("stata_mfx kind must be :dydx, :eyex, :eydx, or :dyex; got $kind")
    title_prefix = kind == :dydx ? "Marginal effects" : "Elasticities"
    kind_label   = kind == :dydx ? "dy/dx" :
                   kind == :eyex ? "ey/ex" :
                   kind == :eydx ? "ey/dx" : "dy/ex"

    n, k = size(X)
    x_eval = at !== nothing ? collect(at) :
             evaluate == :atmean ? vec(Statistics.mean(X, dims=1)) :
             error("stata_mfx supports evaluate=:atmean or `at=<vector>` only")
    μ_at = exp(LinearAlgebra.dot(x_eval, β))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)

    _g(x; sig::Int=7, cap::Int=0) = begin
        (ismissing(x) || !isfinite(x)) && return "."
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        if cap > 0
            while length(s) > cap && sig_use > 1
                sig_use -= 1
                s = Printf.@sprintf("%.*g", sig_use, x)
            end
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        s
    end

    rows = Tuple[]
    for (nm, idx) in factors
        x0 = copy(x_eval); x0[idx] = 0.0
        x1 = copy(x_eval); x1[idx] = 1.0
        μ0 = exp(LinearAlgebra.dot(x0, β))
        μ1 = exp(LinearAlgebra.dot(x1, β))
        val  = μ1 - μ0
        gvec = μ1 .* x1 .- μ0 .* x0
        se   = sqrt(gvec' * V * gvec)
        push!(rows, ("$nm*", val, se, x_eval[idx]))
    end
    for (nm, idx) in continuous
        if kind == :dydx
            val  = β[idx] * μ_at
            gvec = zeros(k); gvec[idx] = μ_at
            gvec .+= β[idx] * μ_at .* x_eval
        elseif kind == :eyex
            val  = β[idx] * x_eval[idx]
            gvec = zeros(k); gvec[idx] = x_eval[idx]
        elseif kind == :eydx
            val  = β[idx]
            gvec = zeros(k); gvec[idx] = 1.0
        else  # :dyex
            xj   = x_eval[idx]
            val  = β[idx] * μ_at * xj
            gvec = zeros(k); gvec[idx] = μ_at * xj
            gvec .+= β[idx] * μ_at * xj .* x_eval
        end
        se = sqrt(gvec' * V * gvec)
        push!(rows, (string(nm), val, se, x_eval[idx]))
    end

    println()
    Printf.@printf("%s after %s\n", title_prefix, model_name)
    Printf.@printf("      y  = %s\n", expression)
    Printf.@printf("         =  %.8g\n", μ_at)
    println("-"^78)
    Printf.@printf("variable |      %s    Std. err.     z    P>|z|  [    95%% C.I.   ]      X\n",
                   kind_label)
    println("-"^9, "+", "-"^68)
    for (nm, val, se, xj) in rows
        z = val / se
        p = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        lo, hi = val - crit*se, val + crit*se
        Printf.@printf("%8s |%11s%12s%8.2f%8.3f%10s%9s%10s\n",
                       nm,
                       lpad(_g(val; sig=7, cap=9), 11),
                       lpad(_g(se;  sig=7, cap=7), 12),
                       z, p,
                       lpad(_g(lo;  sig=7, cap=8), 10),
                       lpad(_g(hi;  sig=7, cap=8),  9),
                       lpad(_g(xj;  sig=7, cap=7), 10))
    end
    println("-"^78)
    !isempty(factors) &&
        println("(*) $kind_label is for discrete change of dummy variable from 0 to 1")
    return rows
end
