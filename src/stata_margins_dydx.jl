# ============================================================================
# stata_margins_dydx.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_margins_dydx(β, V, X; factors, continuous, kind=:dydx,
                       atmean=false, at=nothing, at_labels=nothing,
                       noatlegend=false, interactions=NamedTuple[],
                       level=0.95, vce_label="Robust",
                       expression="Predicted number of events, predict()")

Unified Stata-style `margins, dydx(*)` for exp-link (Poisson) models:

  - default (`atmean=false, at=nothing`) → **AME** ("Average marginal effects").
  - `atmean=true` → **MEM** ("Conditional marginal effects" at the sample mean).
  - `at=c` (length-k vector) → **MER** ("Conditional marginal effects" at c);
    pass `at_labels = [:var=>v, …]` or `noatlegend=true`.

`kind` ∈ (`:dydx`, `:eyex`, `:eydx`, `:dyex`) selects the header and the
formula for *continuous* variables (factors always use the discrete change).
`factors` / `continuous` are vectors of `(name, col_idx::Int)`. `interactions`
entries `(col=k, base=[i,j,…])` mark product columns so factor flips rebuild
them and the chain rule picks up the extra term. Returns the row tuples.
"""
function stata_margins_dydx(β::AbstractVector, V::AbstractMatrix,
                            X::AbstractMatrix;
                            factors::AbstractVector,
                            continuous::AbstractVector,
                            kind::Symbol=:dydx,
                            atmean::Bool=false,
                            at::Union{Nothing,AbstractVector}=nothing,
                            at_labels::Union{Nothing,AbstractVector}=nothing,
                            noatlegend::Bool=false,
                            interactions::AbstractVector=NamedTuple[],
                            level::Float64=0.95,
                            vce_label::String="Robust",
                            expression::String="Predicted number of events, predict()")
    kind in (:dydx, :eyex, :eydx, :dyex) ||
        error("kind must be :dydx, :eyex, :eydx, or :dyex; got $kind")
    kind_label = kind == :dydx ? "dy/dx" :
                 kind == :eyex ? "ey/ex" :
                 kind == :eydx ? "ey/dx" : "dy/ex"
    n, k = size(X)

    g9 = function(x; w_::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w_)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        lpad(s, w_)
    end
    g_atval = function(x; sig::Int=7)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
        s
    end
    commafmt(num::Integer) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    _summary(dydx, se) = begin
        z = dydx / se
        p = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        (z, p, dydx - crit*se, dydx + crit*se)
    end

    if at !== nothing
        mode  = :atpoint
        x_eval = collect(at)
        length(x_eval) == k ||
            error("at must have length(β)=$k, got $(length(x_eval))")
    elseif atmean
        mode  = :atmean
        x_eval = vec(Statistics.mean(X, dims = 1))
    else
        mode  = :average
        x_eval = nothing
    end

    title = mode == :average ? "Average marginal effects" :
                               "Conditional marginal effects"

    function _refresh_interactions!(Xm::AbstractMatrix, idx::Integer)
        for intr in interactions
            if idx in intr.base
                col = ones(size(Xm, 1))
                for b in intr.base
                    col .*= Xm[:, b]
                end
                Xm[:, intr.col] .= col
            end
        end
    end
    function _refresh_interactions_vec!(xv::AbstractVector, idx::Integer)
        for intr in interactions
            if idx in intr.base
                v = 1.0
                for b in intr.base
                    v *= xv[b]
                end
                xv[intr.col] = v
            end
        end
    end

    rows = Tuple[]
    for (nm, idx) in factors
        if mode == :average
            X0 = copy(X); X0[:, idx] .= 0.0; _refresh_interactions!(X0, idx)
            X1 = copy(X); X1[:, idx] .= 1.0; _refresh_interactions!(X1, idx)
            μ0 = exp.(X0 * β); μ1 = exp.(X1 * β)
            dydx = Statistics.mean(μ1 .- μ0)
            gvec = vec((μ1' * X1 .- μ0' * X0) ./ n)
        else
            x0 = copy(x_eval); x0[idx] = 0.0; _refresh_interactions_vec!(x0, idx)
            x1 = copy(x_eval); x1[idx] = 1.0; _refresh_interactions_vec!(x1, idx)
            μ0 = exp(LinearAlgebra.dot(x0, β))
            μ1 = exp(LinearAlgebra.dot(x1, β))
            dydx = μ1 - μ0
            gvec = μ1 .* x1 .- μ0 .* x0
        end
        se = sqrt(gvec' * V * gvec)
        push!(rows, ("1.$nm", dydx, se, _summary(dydx, se)...))
    end
    function _intr_extras(idx::Integer, X_::AbstractMatrix)
        out = NamedTuple[]
        for intr in interactions
            if idx in intr.base
                po = ones(size(X_, 1))
                for b in intr.base
                    b == idx && continue
                    po .*= X_[:, b]
                end
                push!(out, (col = intr.col, prod_other = po))
            end
        end
        out
    end
    function _intr_extras_vec(idx::Integer, xv::AbstractVector)
        out = NamedTuple[]
        for intr in interactions
            if idx in intr.base
                v = 1.0
                for b in intr.base
                    b == idx && continue
                    v *= xv[b]
                end
                push!(out, (col = intr.col, prod_other_at = v))
            end
        end
        out
    end

    for (nm, idx) in continuous
        if mode == :average
            μ_all = exp.(X * β)
            μ̄    = Statistics.mean(μ_all)
            x̄_j  = Statistics.mean(X[:, idx])
            if kind == :dydx
                xtras = _intr_extras(idx, X)
                slope = fill(β[idx], n)
                for x in xtras
                    slope .+= β[x.col] .* x.prod_other
                end
                dydx = Statistics.mean(μ_all .* slope)
                gvec = vec((μ_all .* slope)' * X) ./ n
                gvec[idx] += sum(μ_all) / n
                for x in xtras
                    gvec[x.col] += sum(μ_all .* x.prod_other) / n
                end
            elseif kind == :eyex
                dydx = β[idx] * x̄_j
                gvec = zeros(k); gvec[idx] = x̄_j
            elseif kind == :eydx
                dydx = β[idx]
                gvec = zeros(k); gvec[idx] = 1.0
            else  # :dyex
                dydx = β[idx] * μ̄ * x̄_j
                gvec = zeros(k); gvec[idx] = μ̄ * x̄_j
                gvec .+= β[idx] * x̄_j .* vec((μ_all' * X) ./ n)
            end
        else
            μ_at = exp(LinearAlgebra.dot(x_eval, β))
            xj   = x_eval[idx]
            if kind == :dydx
                xtras = _intr_extras_vec(idx, x_eval)
                slope_at = β[idx]
                for x in xtras
                    slope_at += β[x.col] * x.prod_other_at
                end
                dydx = μ_at * slope_at
                gvec = μ_at * slope_at .* x_eval
                gvec[idx] += μ_at
                for x in xtras
                    gvec[x.col] += μ_at * x.prod_other_at
                end
            elseif kind == :eyex
                dydx = β[idx] * xj
                gvec = zeros(k); gvec[idx] = xj
            elseif kind == :eydx
                dydx = β[idx]
                gvec = zeros(k); gvec[idx] = 1.0
            else  # :dyex
                dydx = β[idx] * μ_at * xj
                gvec = zeros(k); gvec[idx] = μ_at * xj
                gvec .+= β[idx] * μ_at * xj .* x_eval
            end
        end
        se = sqrt(gvec' * V * gvec)
        push!(rows, (string(nm), dydx, se, _summary(dydx, se)...))
    end

    dydx_names = vcat(["1.$nm" for (nm, _) in factors],
                      [string(nm) for (nm, _) in continuous])

    println()
    Printf.@printf("%-57s%-13s = %5s\n",
                   title, "Number of obs", commafmt(n))
    Printf.@printf("Model VCE: %s\n", vce_label)
    println()
    Printf.@printf("Expression: %s\n", expression)
    Printf.@printf("%s wrt:  %s\n", kind_label, join(dydx_names, " "))

    if !noatlegend && mode != :average
        if mode == :atmean
            at_lines = Tuple[]
            for (nm, idx) in factors
                m1 = Statistics.mean(X[:, idx])
                push!(at_lines, ("0.$nm", 1.0 - m1))
                push!(at_lines, ("1.$nm", m1))
            end
            for (nm, idx) in continuous
                push!(at_lines, (string(nm), x_eval[idx]))
            end
            max_name_w = maximum(length(t[1]) for t in at_lines)
            for (i, (nm, val)) in enumerate(at_lines)
                prefix = i == 1 ? "At: " : "    "
                Printf.@printf("%s%-*s = %s (mean)\n",
                               prefix, max_name_w, nm, g_atval(val))
            end
        elseif at_labels !== nothing && !isempty(at_labels)
            max_name_w = maximum(length(string(p_.first)) for p_ in at_labels)
            val_strs   = [Printf.@sprintf("%g", float(p_.second)) for p_ in at_labels]
            max_val_w  = maximum(length, val_strs)
            for (i, p_) in enumerate(at_labels)
                prefix = i == 1 ? "At: " : "    "
                Printf.@printf("%s%-*s = %s\n",
                               prefix, max_name_w, string(p_.first),
                               lpad(val_strs[i], max_val_w))
            end
        end
    end

    println()
    println("-"^78)
    println("             |            Delta-method")
    Printf.@printf("             |      %s   std. err.      z    P>|z|     [95%% conf. interval]\n", kind_label)
    println("-"^13, "+", "-"^64)
    for (lab, dydx, se, z, p, lo, hi) in rows
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       lab, g9(dydx; w_=10), g9(se; w_=9),
                       Printf.@sprintf("%7.2f", z),
                       Printf.@sprintf("%.3f", p),
                       g9(lo; w_=9), g9(hi; w_=10))
    end
    println("-"^78)
    !isempty(factors) && println("Note: dy/dx for factor levels is the discrete change from the base level.")
    return rows
end
