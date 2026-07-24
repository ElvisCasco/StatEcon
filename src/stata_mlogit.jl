import Optim
import ForwardDiff

# === ch15 shared helpers (defined in this primary multinomial-models file;
#     referenced at runtime by the sibling stata_*.jl ch15 files) ===

# Unwrap a ReadStatTables.LabeledValue to its underlying value.
_c15_raw(x) = hasproperty(x, :value) ? x.value : x

# erf via the standard-normal cdf (SpecialFunctions is unavailable):
#   erf(z) = 2Φ(z√2) − 1  ⇒  0.5(1 + erf(x/√2)) = Φ(x).
# ForwardDiff-differentiable (Distributions' normal cdf carries Duals).
_c15_erf(z) = 2 * Distributions.cdf(Distributions.Normal(), z * sqrt(2.0)) - 1

# L-BFGS with an explicit ForwardDiff gradient. Optim's `autodiff = :forward`
# Symbol pipeline is broken in the pinned version, so we build the gradient
# ourselves and call the 5-arg optimize. A stray `autodiff` kwarg is ignored.
function _c15_optimize(f, x0, method, opts = Optim.Options(); kwargs...)
    g!(G, x) = ForwardDiff.gradient!(G, f, x)
    return Optim.optimize(f, g!, x0, method, opts)
end

"""
    stata_mlogit(df, depvar, regs; baseoutcome=nothing, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `mlogit <depvar> <regs>, baseoutcome(<v>) nolog`. Fits a
multinomial logistic regression by ML (`Optim.LBFGS` with autodiff
gradient). `baseoutcome` is the raw value of the reference category —
defaults to the smallest sorted level.

Internal parameterisation: design `X = [1, regs...]` (intercept first,
to match downstream chunks). Parameter matrix `B` is `K × (J-1)`,
column `j` carrying the log-odds coefficients for the `j`-th non-base
category in sorted order.

OIM vcov via finite-difference Hessian. Output matches Stata: header
(Number of obs / LR chi2(p) / Prob > chi2 / Log likelihood / Pseudo
R2) then per-category coefficient blocks with the base outcome row
marked `(base outcome)`.

Returns `(; B, V, se, X, J, dfc, depvar_name, regs, base_value,
cat_labels, ll, ll_null, LR, LR_p, pseudo_r2, n)`.
"""
function stata_mlogit(df::DataFrames.AbstractDataFrame, depvar::Symbol,
                     regs::AbstractVector{Symbol};
                     baseoutcome = nothing,
                     level::Float64 = 0.95,
                     quiet::Bool = false,
                     rrr::Bool = false)
    cols = vcat([depvar], collect(regs))
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c15_raw.(col))
        end
    end

    y_raw = [_c15_raw(v) for v in dfc[!, depvar]]
    cats  = sort(unique(y_raw))
    J     = length(cats)
    base_val = baseoutcome === nothing ? cats[1] : baseoutcome
    base_pos = findfirst(==(base_val), cats)
    base_pos === nothing && error("baseoutcome $base_val not in $depvar")
    others_pos = [j for j in 1:J if j != base_pos]

    # Label per category (string label if labeled, else stringified value)
    cat_labels = String[]
    for (i, c) in enumerate(cats)
        idx = findfirst(==(c), y_raw)
        v   = dfc[idx, depvar]
        push!(cat_labels,
              hasproperty(v, :labels) && hasproperty(v, :value) &&
              haskey(v.labels, v.value) ? v.labels[v.value] : string(c))
    end

    y_idx = [findfirst(==(yi), cats) for yi in y_raw]
    N = length(y_idx)
    X = hcat(ones(N), [Float64.(_c15_raw.(dfc[!, r])) for r in regs]...)
    K = size(X, 2)
    coefnames = ["_cons", string.(regs)...]

    function negll(θ)
        B = reshape(θ, K, J - 1)
        η = X * B                              # N × (J-1)
        ll = zero(eltype(θ))
        for i in 1:N
            den = one(eltype(θ))
            for j in 1:(J-1); den += exp(η[i, j]); end
            if y_idx[i] == base_pos
                ll -= log(den)
            else
                k = findfirst(==(y_idx[i]), others_pos)
                ll += η[i, k] - log(den)
            end
        end
        return -ll
    end

    θ0 = zeros(K * (J - 1))
    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 1000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    B  = reshape(θ̂, K, J - 1)
    ll = -Optim.minimum(res)

    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x)
            xpi[i] += h_[i]; xmi[i] -= h_[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h_[i]^2
            for j in (i+1):nθ
                xpp = copy(x); xpp[i] += h_[i]; xpp[j] += h_[j]
                xpm = copy(x); xpm[i] += h_[i]; xpm[j] -= h_[j]
                xmp = copy(x); xmp[i] -= h_[i]; xmp[j] += h_[j]
                xmm = copy(x); xmm[i] -= h_[i]; xmm[j] -= h_[j]
                H[i, j] = H[j, i] =
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h_[i] * h_[j])
            end
        end
        return H
    end
    H = _fd_hessian(negll, θ̂)
    V = LinearAlgebra.inv(LinearAlgebra.Symmetric(H))
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    SE = reshape(se, K, J - 1)

    counts = [count(==(j), y_idx) for j in 1:J]
    ll_null = sum(c * log(c / N) for c in counts if c > 0)
    LR = 2 * (ll - ll_null)
    df_lr = (K - 1) * (J - 1)                  # only slope coefficients
    LR_p = 1 - Distributions.cdf(Distributions.Chisq(df_lr), LR)
    pseudo_r2 = 1 - ll / ll_null

    if !quiet
        crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
        function g9(x; w::Int = 10, sig::Int = 7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su = sig
            s = Printf.@sprintf("%.*g", su, x)
            cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
            while length(s) > cap && su > 1
                su -= 1
                s = Printf.@sprintf("%.*g", su, x)
            end
            0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
            lpad(s, w)
        end
        commafmt(num) = begin
            s = string(abs(num)); parts = String[]; i = length(s)
            while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
            (num < 0 ? "-" : "") * join(reverse(parts), ",")
        end

        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Multinomial logistic regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6.2f\n", "",
                       "LR chi2($(df_lr))", LR)
        Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", LR_p)
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad, r2_str)
        println()

        println("-"^78)
        # `rrr` swaps the point-estimate column header to "RRR" (relative-
        # risk ratios = exp(β)). z and p are unchanged (test is still β = 0
        # ⇔ RRR = 1); SE uses the delta method, SE(exp β) = exp(β)·SE(β);
        # CI endpoints are exp(β ± crit·SE(β)).
        coef_hdr = rrr ? "       RRR" : "Coefficient"
        Printf.@printf("%12s | %s  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), coef_hdr, 100 * level)
        println("-"^13, "+", "-"^64)

        # Stata prints each outcome's block in sorted order; base goes first.
        for (i, j) in enumerate(1:J)
            Printf.@printf("%-12s |", cat_labels[j])
            println()
            if j == base_pos
                Printf.@printf("%12s |  (base outcome)\n", "")
            else
                k = findfirst(==(j), others_pos)
                # Stata prints slope rows first, then _cons.
                ord = vcat(2:K, [1])
                for r in ord
                    label = r == 1 ? "_cons" : coefnames[r]
                    b  = B[r, k]; s = SE[r, k]
                    z  = b / s
                    p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
                    if rrr
                        rr_pt = exp(b)
                        rr_se = rr_pt * s        # delta method
                        rr_lo = exp(b - crit * s)
                        rr_hi = exp(b + crit * s)
                        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                                       label, g9(rr_pt; w = 10), g9(rr_se; w = 9),
                                       Printf.@sprintf("%7.2f", z),
                                       Printf.@sprintf("%.3f", p),
                                       g9(rr_lo; w = 9), g9(rr_hi; w = 10))
                    else
                        lo = b - crit * s
                        hi = b + crit * s
                        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                                       label, g9(b; w = 10), g9(s; w = 9),
                                       Printf.@sprintf("%7.2f", z),
                                       Printf.@sprintf("%.3f", p),
                                       g9(lo; w = 9), g9(hi; w = 10))
                    end
                end
            end
            i < J && println("-"^13, "+", "-"^64)
        end
        println("-"^78)
        Printf.@printf("%s outcome = %s (%s)\n", "Base", string(base_val),
                       cat_labels[base_pos])
    end

    return (; B, V, se, SE, X, J, dfc,
              depvar_name = depvar, regs = collect(regs),
              base_value = base_val, base_pos, cat_labels,
              ll, ll_null, LR, LR_p, pseudo_r2,
              n = N, nparam = K * (J - 1))
end

"""
    _mnl_probs(B, X, J) -> Matrix

Stata `predict, pr` for multinomial logit. Closed-form softmax with
the base outcome's log-odds pinned at 0:

    P[i, base] = 1 / (1 + Σⱼ exp(Xᵢ · Bⱼ))
    P[i, j]    = exp(Xᵢ · Bⱼ) / (1 + Σⱼ exp(Xᵢ · Bⱼ))   for j ≠ base

`B` is the `K × (J-1)` coefficient matrix returned by `stata_mlogit`
(column k = the k-th non-base outcome in sorted order; rows match `X`
which already includes the intercept column). Always assumes the
*first* category is the base — matches the chunk-1 mlogit fit that
sets `baseoutcome = cats[1]`.
"""
function _mnl_probs(B, X, J)
    eta = X * B
    den = 1.0 .+ sum(exp.(eta); dims = 2)
    P   = zeros(size(X, 1), J)
    P[:, 1] = 1.0 ./ den
    for j in 2:J
        P[:, j] = exp.(eta[:, j - 1]) ./ den
    end
    return P
end

"""
    stata_test_mlogit(res, vars) -> NamedTuple

Stata `test <var>` post-`mlogit`: joint Wald test that the coefficient
on each named regressor equals 0 across **all** outcome equations.
`res` is the NamedTuple returned by `stata_mlogit`; `vars` is a Symbol
or `Vector{Symbol}` of regressor names already in the fitted model.

Stata lists one constraint per (outcome × variable) — the base
outcome's row carries the `o.` (omitted) marker — then reports the
dropped base constraint and the resulting `chi2(dof)` / `Prob > chi2`
on the remaining `(J-1) × length(vars)` restrictions.

Returns `(; chi2, dof, p, R)` (R = restriction matrix in the layout
of `vec(B)` from the fit).

!!! note "Limitation"
    Only the all-equations form (Stata `test <vars>`) is implemented — the
    joint test that each named regressor is zero across every outcome
    equation. The equation-specific restriction (`test [k]: <vars>`, one
    outcome at a time) is not supported; build that restriction matrix by
    hand from `res.B` / `res.V` — see Example 15.4 in
    `examples/Wooldridge_Panel/ch15` for the pattern.
"""
function stata_test_mlogit(res, vars)
    vs = vars isa Symbol ? [vars] : collect(vars)
    K = size(res.X, 2); J = res.J
    base_pos = res.base_pos
    cat_labels = res.cat_labels
    regs = collect(res.regs)
    coef_rows = Dict(v => i + 1 for (i, v) in enumerate(regs))   # row in B (row 1 = _cons)

    R_rows = Vector{Vector{Float64}}()
    constraint_lines = String[]
    counter = 1
    for v in vs
        haskey(coef_rows, v) || error("variable $v not in mlogit regressors")
        rb = coef_rows[v]
        for j in 1:J
            marker = j == base_pos ? "o." : ""
            push!(constraint_lines,
                  Printf.@sprintf(" (%2d)  [%s]%s%s = 0",
                                  counter, cat_labels[j], marker, string(v)))
            counter += 1
            if j != base_pos
                k = j > base_pos ? j - 1 : j   # column in B for non-base outcome j
                row = zeros(K * (J - 1))
                row[(k - 1) * K + rb] = 1.0
                push!(R_rows, row)
            end
        end
    end

    R   = reduce(vcat, [r' for r in R_rows])
    βv  = vec(res.B)
    Rβ  = R * βv
    χ²  = Rβ' * (R * res.V * R' \ Rβ)
    dof = size(R, 1)
    p   = 1 - Distributions.cdf(Distributions.Chisq(dof), χ²)

    for line in constraint_lines; println(line); end
    Printf.@printf("       Constraint %d dropped\n", base_pos)
    println()
    Printf.@printf("           chi2(%3d) = %7.2f\n", dof, χ²)
    Printf.@printf("         Prob > chi2 = %9.4f\n", p)

    return (; chi2 = χ², dof, p, R)
end

"""
    stata_margins_mlogit(res; outcome, level=0.95, quiet=false) -> NamedTuple

Stata `margins, predict(outcome(<k>)) noatlegend` after `mlogit`. The
margin is the *sample-average* predicted probability of the requested
outcome,  P̄ = (1/N) Σᵢ P̂ᵢ(outcome), with delta-method SE built from
the per-observation gradient of `P̂ᵢ(outcome)` w.r.t. the full vec(B)
parameter vector and `res.V`.

Gradient (closed form for MNL with base outcome `b` and β_b ≡ 0):

    ∂Pᵢ(j) / ∂β_l = Xᵢ · Pᵢ(j) · (δ_jl − Pᵢ(l))      l ≠ b

Stacked over the K × (J-1) layout of `B`, then averaged over `i`,
gives a length-`K(J-1)` gradient vector `g`; SE = √(gᵀ · V · g).

Returns `(; margin, se, z, p, ci_lo, ci_hi, target_outcome,
target_label, n)`.
"""
function stata_margins_mlogit(res; outcome,
                              level::Float64 = 0.95,
                              quiet::Bool = false)
    K = size(res.X, 2); J = res.J
    base_pos = res.base_pos
    others_pos = [j for j in 1:J if j != base_pos]

    # Locate the target outcome (Stata's `outcome(<k>)` accepts either
    # the *raw* category value — e.g. 3 — or the 1-based sorted index;
    # we accept both and disambiguate.)
    tgt_pos = nothing
    if outcome isa Integer && 1 <= outcome <= J &&
       (outcome == base_pos || outcome ∉ Set(res.cat_labels))
        tgt_pos = outcome   # treat as sorted-index when in [1, J]
    end
    if tgt_pos === nothing
        # Try matching against the raw category value via cat_labels
        # (LabeledValue-style) or fall back to the value itself.
        # Recover the underlying sorted raw values from res.dfc.
        raws = sort(unique(_c15_raw.(res.dfc[!, res.depvar_name])))
        tgt_pos = findfirst(==(outcome), raws)
        tgt_pos === nothing && error("outcome=$outcome not in $(res.depvar_name)")
    end
    target_label = res.cat_labels[tgt_pos]

    # Predicted probabilities (chunk-1 _mnl_probs).
    P = _mnl_probs(res.B, res.X, J)
    N = size(P, 1)

    # Sample-average margin.
    margin = sum(P[:, tgt_pos]) / N

    # Gradient w.r.t. each non-base coefficient.
    # G[r, k]  =  (1/N) Σᵢ Xᵢ_r · Pᵢ(tgt) · (δ_{others_pos[k], tgt} − Pᵢ(others_pos[k]))
    G = zeros(K, J - 1)
    for k in 1:(J - 1)
        ko = others_pos[k]
        δ  = ko == tgt_pos ? 1.0 : 0.0
        # contribution = Pᵢ(tgt) · (δ − Pᵢ(ko)) — column vector length N
        w = P[:, tgt_pos] .* (δ .- P[:, ko])
        G[:, k] = res.X' * w ./ N
    end
    g = vec(G)
    se = sqrt(max(g' * res.V * g, 0.0))
    z  = margin / se
    p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    ci_lo = margin - crit * se
    ci_hi = margin + crit * se

    if !quiet
        function g9(x; w::Int = 10, sig::Int = 7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su = sig
            s = Printf.@sprintf("%.*g", su, x)
            cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
            while length(s) > cap && su > 1
                su -= 1
                s = Printf.@sprintf("%.*g", su, x)
            end
            0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
            lpad(s, w)
        end
        commafmt(num) = begin
            s = string(abs(num)); parts = String[]; i = length(s)
            while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
            (num < 0 ? "-" : "") * join(reverse(parts), ",")
        end
        println()
        Printf.@printf("%-56s%-13s = %6s\n", "Predictive margins",
                       "Number of obs", commafmt(N))
        println("Model VCE: OIM")
        Printf.@printf("Expression: Pr(%s==%s), predict(outcome(%s))\n",
                       string(res.depvar_name), target_label, string(outcome))
        println()
        println("-"^78)
        println("             |            Delta-method")
        Printf.@printf("%12s |     Margin   Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100 * level)
        println("-"^13, "+", "-"^64)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "_cons", g9(margin; w = 10), g9(se; w = 9),
                       Printf.@sprintf("%7.2f", z),
                       Printf.@sprintf("%.3f", p),
                       g9(ci_lo; w = 9), g9(ci_hi; w = 10))
        println("-"^78)
    end

    return (; margin, se, z, p, ci_lo, ci_hi,
              target_outcome = tgt_pos, target_label, n = N)
end

"""
    stata_margins_dydx_mlogit(res; outcome, dydx=:all, at=:mean,
                              level=0.95, quiet=false) -> NamedTuple

Stata `margins, dydx(*) predict(outcome(<k>))` after `mlogit`. Closed-
form MNL partial derivative

    ∂Pᵢ(outcome) / ∂x_c   = Pᵢ(outcome) · (β_{outcome,c} − Σₗ β_{l,c}·Pᵢ(l))

reduced to a single scalar per regressor by one of three policies:

* `at = :mean`        — evaluate at the column means of `res.X`
                       (Stata's `atmean`, MEM = marginal effect at the
                        mean)
* `at = :asobserved`  — evaluate at every row of `res.X` and average
                       (Stata's default, AME = average marginal effect)
* numeric `Vector{Float64}` of length `K` — evaluate at that single
                       anchor point (constant in position 1)

SE is delta-method on the closed-form gradient above, plugged through
a numerical Jacobian w.r.t. `vec(B)` and `res.V`.

`dydx = :all` covers every regressor in `res.regs`; pass a Symbol /
Vector of Symbols to restrict.

Returns `(; rows, target_outcome, target_label, at, n)` where `rows`
is a Vector of NamedTuples `(reg, dydx, se, z, p, ci_lo, ci_hi)`.
"""
function stata_margins_dydx_mlogit(res; outcome,
                                   dydx = :all,
                                   at = :mean,
                                   level::Float64 = 0.95,
                                   quiet::Bool = false)
    K = size(res.X, 2); J = res.J; base_pos = res.base_pos
    others_pos = [j for j in 1:J if j != base_pos]
    raws = sort(unique(_c15_raw.(res.dfc[!, res.depvar_name])))
    tgt_pos = findfirst(==(outcome), raws)
    tgt_pos === nothing && error("outcome=$outcome not in $(res.depvar_name)")
    target_label = res.cat_labels[tgt_pos]

    # Evaluation matrix: each row is one anchor point in X-space. For
    # `:mean` and numeric `at` we get a single row; for `:asobserved`
    # we use the full estimation sample.
    Xeval = at === :mean       ? reshape(vec(Statistics.mean(res.X; dims = 1)),
                                         1, K) :
            at === :asobserved ? res.X :
            reshape(convert(Vector{Float64}, at), 1, K)
    size(Xeval, 2) == K || error("`at` must have length $K (got $(length(at)))")

    # Which regressors to take dy/dx over. res.regs lists the user
    # regressors in order (index 1 of res.regs = column 2 of X, since
    # X's column 1 is the constant).
    regs_all = collect(res.regs)
    sel = dydx === :all                     ? regs_all :
          dydx isa Symbol                   ? [dydx]   :
          collect(dydx)
    reg_cols = Int[]
    for r in sel
        i = findfirst(==(r), regs_all)
        i === nothing && error("regressor $r not in res.regs")
        push!(reg_cols, i + 1)   # +1 because constant is column 1
    end

    # mem_at(theta, c): average MEM_c over the rows of `Xeval`. With a
    # single-row Xeval this collapses to the at-mean / at-anchor MEM;
    # with the full sample it gives the AME.
    function mem_at(theta, c)
        B = reshape(theta, K, J - 1)
        Nrows = size(Xeval, 1)
        acc = zero(eltype(theta))
        β_c = zeros(eltype(theta), J)
        for k in 1:(J - 1); β_c[others_pos[k]] = B[c, k]; end
        for i in 1:Nrows
            xi = view(Xeval, i, :)
            η  = [LinearAlgebra.dot(xi, B[:, j]) for j in 1:(J - 1)]
            m  = maximum(η; init = 0.0)
            den = exp(-m) + sum(exp.(η .- m))
            P = zeros(eltype(theta), J)
            P[base_pos] = exp(-m) / den
            for k in 1:(J - 1)
                P[others_pos[k]] = exp(η[k] - m) / den
            end
            acc += P[tgt_pos] * (β_c[tgt_pos] - LinearAlgebra.dot(β_c, P))
        end
        return acc / Nrows
    end

    θ̂ = vec(res.B)
    h_step = sqrt(eps(Float64))

    crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    rows = NamedTuple[]
    for (rname, c) in zip(sel, reg_cols)
        m_pt = mem_at(θ̂, c)
        # Central-FD gradient w.r.t. each θ entry.
        g = zeros(length(θ̂))
        for i in eachindex(θ̂)
            step = max(abs(θ̂[i]) * h_step, h_step)
            tp = copy(θ̂); tp[i] += step
            tm = copy(θ̂); tm[i] -= step
            g[i] = (mem_at(tp, c) - mem_at(tm, c)) / (2 * step)
        end
        se = sqrt(max(g' * res.V * g, 0.0))
        z  = m_pt / se
        p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
        push!(rows, (; reg = rname, dydx = m_pt, se, z, p,
                      ci_lo = m_pt - crit * se, ci_hi = m_pt + crit * se))
    end

    if !quiet
        function g9(x; w::Int = 10, sig::Int = 7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su = sig
            s = Printf.@sprintf("%.*g", su, x)
            cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
            while length(s) > cap && su > 1
                su -= 1
                s = Printf.@sprintf("%.*g", su, x)
            end
            0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
            lpad(s, w)
        end
        commafmt(num) = begin
            s = string(abs(num)); parts = String[]; i = length(s)
            while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
            (num < 0 ? "-" : "") * join(reverse(parts), ",")
        end
        N = size(res.X, 1)
        println()
        # Stata distinguishes "Conditional marginal effects" (at a
        # specific anchor like atmean) from "Average marginal effects"
        # (averaging over the estimation sample).
        title = at === :asobserved ? "Average marginal effects" :
                                     "Conditional marginal effects"
        Printf.@printf("%-56s%-13s = %6s\n", title,
                       "Number of obs", commafmt(N))
        println("Model VCE: OIM")
        Printf.@printf("Expression: Pr(%s==%s), predict(outcome(%s))\n",
                       string(res.depvar_name), target_label, string(outcome))
        Printf.@printf("dy/dx wrt: %s\n", join(string.(sel), " "))
        if at === :mean
            xb = vec(Xeval[1, :])
            atstr = join((string(r) * " = " *
                          Printf.@sprintf("%.6g", xb[i + 1]) * " (mean)"
                          for (i, r) in enumerate(regs_all)), "  ")
            println("At: ", atstr)
        end
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

    return (; rows, target_outcome = tgt_pos, target_label,
              at = Xeval, n = size(res.X, 1))
end
