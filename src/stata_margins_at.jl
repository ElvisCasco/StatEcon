# ============================================================================
# stata_margins_at.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_margins_at(β, V, X; at=nothing, at_labels=nothing,
                     over=nothing, over_levels=nothing,
                     mean_fn=:exp, vce_label="Robust", level=0.95,
                     expression="Predicted number of events, predict()",
                     name="_cons")

Stata-style `margins` with delta-method SE. Three modes:

  - `at = nothing, over = nothing` → **Predictive margins** (sample average of
    `predict` over X).
  - `at = c` (length-k vector incl. the constant slot) → **Adjusted
    predictions** at the single point. Pass `at_labels = [:var => v, …]` for
    the "At:" block.
  - `over = "var" => idx` → **Predictive margins by factor level**: set
    `X[:, idx] .= k` for all observations, one row per level.

`mean_fn` ∈ (`:exp`, `:identity`, `:logit`, `:probit`). Pass the
n/(n−1)-corrected robust V to match Stata's `margins` after `poisson,
vce(robust)`. Returns the row tuple(s).
"""
function stata_margins_at(β::AbstractVector, V::AbstractMatrix, X::AbstractMatrix;
                       at::Union{Nothing,AbstractVector}=nothing,
                       at_labels::Union{Nothing,AbstractVector}=nothing,
                       over::Union{Nothing,Pair}=nothing,
                       over_levels::Union{Nothing,AbstractVector}=nothing,
                       mean_fn::Symbol=:exp, vce_label::String="Robust",
                       level::Float64=0.95, name::String="_cons",
                       expression::String="Predicted number of events, predict()")
    n_sample, _ = size(X)

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
    commafmt(num::Integer) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end
    function _summary_row(margin, se)
        z      = margin / se
        p      = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        crit   = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
        (margin, se, z, p, margin - crit*se, margin + crit*se)
    end
    function _compute_margin_at_X(Xm)
        if mean_fn == :exp
            μ    = exp.(Xm * β)
            gvec = vec((μ' * Xm) ./ size(Xm, 1))
            margin = Statistics.mean(μ)
        elseif mean_fn == :identity
            μ    = Xm * β
            gvec = vec(Statistics.mean(Xm, dims = 1))
            margin = Statistics.mean(μ)
        elseif mean_fn == :logit
            η    = Xm * β
            μ    = 1 ./ (1 .+ exp.(-η))
            w    = μ .* (1 .- μ)
            gvec = vec((w' * Xm) ./ size(Xm, 1))
            margin = Statistics.mean(μ)
        elseif mean_fn == :probit
            η    = Xm * β
            μ    = Distributions.cdf.(Distributions.Normal(), η)
            w    = Distributions.pdf.(Distributions.Normal(), η)
            gvec = vec((w' * Xm) ./ size(Xm, 1))
            margin = Statistics.mean(μ)
        else
            error("mean_fn=$mean_fn not supported (use :exp / :identity / :logit / :probit)")
        end
        margin, sqrt(gvec' * V * gvec)
    end

    if over !== nothing
        title = "Predictive margins"
        factor_name, factor_idx = over
        levels_to_use = over_levels === nothing ?
                        sort(unique(X[:, factor_idx])) : collect(over_levels)
        rows = Tuple[]
        for lev in levels_to_use
            Xm = copy(X)
            Xm[:, factor_idx] .= lev
            m, s = _compute_margin_at_X(Xm)
            push!(rows, (lev, _summary_row(m, s)...))
        end
    elseif at === nothing
        title = "Predictive margins"
        m, s  = _compute_margin_at_X(X)
        rows  = [(name, _summary_row(m, s)...)]
    else
        title = "Adjusted predictions"
        c = collect(at)
        length(c) == length(β) ||
            error("at must have length(β)=$(length(β)), got $(length(c))")
        if mean_fn == :exp
            margin = exp(LinearAlgebra.dot(c, β))
            gvec   = margin .* c
        elseif mean_fn == :identity
            margin = LinearAlgebra.dot(c, β)
            gvec   = c
        elseif mean_fn == :logit
            η      = LinearAlgebra.dot(c, β)
            Λ      = 1 / (1 + exp(-η))
            margin = Λ
            gvec   = Λ * (1 - Λ) .* c
        elseif mean_fn == :probit
            η      = LinearAlgebra.dot(c, β)
            margin = Distributions.cdf(Distributions.Normal(), η)
            ϕ      = Distributions.pdf(Distributions.Normal(), η)
            gvec   = ϕ .* c
        else
            error("mean_fn=$mean_fn not supported (use :exp / :identity / :logit / :probit)")
        end
        rows = [(name, _summary_row(margin, sqrt(gvec' * V * gvec))...)]
    end

    println()
    Printf.@printf("%-57s%-13s = %5s\n",
                   title, "Number of obs", commafmt(n_sample))
    Printf.@printf("Model VCE: %s\n", vce_label)
    println()
    Printf.@printf("Expression: %s\n", expression)
    if at_labels !== nothing && !isempty(at_labels)
        max_name_w = maximum(length(string(p_.first)) for p_ in at_labels)
        val_strs   = [Printf.@sprintf("%g", float(p_.second)) for p_ in at_labels]
        max_val_w  = maximum(length, val_strs)
        for (i, p_) in enumerate(at_labels)
            prefix = i == 1 ? "At: " : "    "
            Printf.@printf("%s%-*s = %s\n", prefix, max_name_w,
                           string(p_.first), lpad(val_strs[i], max_val_w))
        end
    end
    println()
    println("-"^78)
    println("             |            Delta-method")
    println("             |     Margin   std. err.      z    P>|z|     [95% conf. interval]")
    println("-"^13, "+", "-"^64)
    if over !== nothing
        Printf.@printf("%12s |\n", first(over))
        for (lev, m, s, z, p, lo, hi) in rows
            lev_str = string(lev isa AbstractFloat && lev == floor(lev) ?
                             Int(lev) : lev)
            Printf.@printf("%11s  | %s  %s  %s   %s    %s  %s\n",
                           lev_str, g9(m; w_=10), g9(s; w_=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", p),
                           g9(lo; w_=9), g9(hi; w_=10))
        end
    else
        for (lab, m, s, z, p, lo, hi) in rows
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           lab, g9(m; w_=10), g9(s; w_=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", p),
                           g9(lo; w_=9), g9(hi; w_=10))
        end
    end
    println("-"^78)
    return rows
end
