# --------------------------------------------------------------------------
# stata_reg3 — Three-stage least squares system estimation.
# Stata equivalent: `reg3 (eq1) (eq2) ...`
# Algorithm: 2SLS per equation → Σ̂ → block GLS with W = Σ̂⁻¹ ⊗ P_Z.
# The instrument set is the union of all exogenous RHS variables (no LHS
# appears as an instrument).
# --------------------------------------------------------------------------

"""
    stata_reg3(df, eqs::Pair...; level=0.95)

Three-stage least-squares system estimator. Each `Pair` is
`:y_j => [x_j1, x_j2, ...]`. All LHS names are treated as endogenous;
any RHS variable that is not itself a LHS is exogenous and enters the
instrument matrix. Returns a NamedTuple with the full β, its vcov, and
per-equation slices (mirrors `stata_sureg_fit`).
"""
function stata_reg3(df::DataFrames.AbstractDataFrame, eqs::Pair...;
                    level::Float64 = 0.95)
    M = length(eqs)

    endog_set = Set{Symbol}()
    for (y, _) in eqs
        push!(endog_set, Symbol(y))
    end
    exog_set = Symbol[]
    seen     = Set{Symbol}()
    for (_, rhs) in eqs
        for v in rhs
            sv = Symbol(v)
            if !(sv in endog_set) && !(sv in seen)
                push!(exog_set, sv); push!(seen, sv)
            end
        end
    end

    all_vars = Symbol[]
    for (y, rhs) in eqs
        push!(all_vars, Symbol(y))
        append!(all_vars, Symbol.(rhs))
    end
    d = DataFrames.dropmissing(df, unique(all_vars))
    N = DataFrames.nrow(d)

    Xs      = Vector{Matrix{Float64}}(undef, M)
    Ys      = Vector{Vector{Float64}}(undef, M)
    cns     = Vector{Vector{String}}(undef, M)
    eqnames = Vector{String}(undef, M)
    for (j, (y, rhs)) in enumerate(eqs)
        ys    = Symbol(y)
        rhs_v = [Symbol(v) for v in rhs]
        Xs[j] = hcat(ones(N),
                     [Float64.(_sm_rawval.(d[!, v])) for v in rhs_v]...)
        Ys[j] = Float64.(_sm_rawval.(d[!, ys]))
        cns[j] = vcat("_cons", string.(rhs_v))
        eqnames[j] = string(ys)
    end
    ks      = [size(X, 2) for X in Xs]
    kt      = sum(ks)
    offsets = vcat(0, cumsum(ks))

    Z    = hcat(ones(N),
                [Float64.(_sm_rawval.(d[!, v])) for v in exog_set]...)
    ZtZi = LinearAlgebra.inv(Z' * Z)
    PX   = [Z * (ZtZi * (Z' * Xs[j])) for j in 1:M]

    # Stage 1: 2SLS per equation → residuals for Σ̂
    β_2sls = Vector{Vector{Float64}}(undef, M)
    e_2sls = Vector{Vector{Float64}}(undef, M)
    for j in 1:M
        β_2sls[j] = (PX[j]' * PX[j]) \ (PX[j]' * Ys[j])
        e_2sls[j] = Ys[j] .- Xs[j] * β_2sls[j]
    end
    Σ̂    = [LinearAlgebra.dot(e_2sls[i], e_2sls[j]) / N for i in 1:M, j in 1:M]
    Σinv = LinearAlgebra.inv(Σ̂)

    # Stage 2: 3SLS block GLS
    XWX = zeros(kt, kt)
    XWY = zeros(kt)
    for i in 1:M, j in 1:M
        ri = (offsets[i]+1):offsets[i+1]
        rj = (offsets[j]+1):offsets[j+1]
        XWX[ri, rj] .= Σinv[i, j] .* (PX[i]' * PX[j])
        XWY[ri]    .+= Σinv[i, j] .* (PX[i]' * Ys[j])
    end
    β_3sls = XWX \ XWY
    V_3sls = LinearAlgebra.inv(XWX)

    β_eq  = [β_3sls[(offsets[j]+1):offsets[j+1]]                              for j in 1:M]
    V_eq  = [V_3sls[(offsets[j]+1):offsets[j+1], (offsets[j]+1):offsets[j+1]] for j in 1:M]
    se_eq = [sqrt.(max.(LinearAlgebra.diag(V), 0.0)) for V in V_eq]
    e_3sls = [Ys[j] .- Xs[j] * β_eq[j] for j in 1:M]

    rmse = [sqrt(LinearAlgebra.dot(e_3sls[j], e_3sls[j]) / N) for j in 1:M]
    r2s   = Float64[]; chi2s = Float64[]; pvals = Float64[]
    for j in 1:M
        rss = LinearAlgebra.dot(e_3sls[j], e_3sls[j])
        tss = sum((Ys[j] .- Statistics.mean(Ys[j])).^2)
        push!(r2s, 1 - rss / tss)
        slope = 2:ks[j]
        βs = β_eq[j][slope]; Vs = V_eq[j][slope, slope]
        χ2 = βs' * (Vs \ βs)
        push!(chi2s, χ2)
        push!(pvals, 1 - Distributions.cdf(Distributions.Chisq(length(slope)), χ2))
    end

    println("Three-stage least-squares regression\n")

    # Summary block (uses stata_sureg's shared table printer)
    sum_headers = ["Equation", "Obs", "Params", "RMSE", "R-squared",
                   "chi2", "P>chi2"]
    sum_rows = Vector{Vector{String}}()
    for j in 1:M
        push!(sum_rows,
              [eqnames[j], string(N), string(ks[j] - 1),
               Printf.@sprintf("%.6f", rmse[j]),
               Printf.@sprintf("%.4f", r2s[j]),
               Printf.@sprintf("%.2f", chi2s[j]),
               Printf.@sprintf("%.4f", pvals[j])])
    end
    _sug_print_table(sum_headers, sum_rows,
                     [:l, :r, :r, :r, :r, :r, :r])
    println()

    # Coefficient block (slopes first, `_cons` last per equation)
    lvl = round(Int, 100 * level)
    zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    coef_headers = ["Equation", "Variable", "Coefficient", "Std. err.",
                    "z", "P>|z|", "[$(lvl)% CI low]", "[$(lvl)% CI high]"]
    rows = Vector{Vector{String}}()
    hline_at = Int[]; nrows = 0
    for j in 1:M
        first_row = true
        order = vcat(collect(2:ks[j]), 1)
        for i in order
            β_i  = β_eq[j][i]
            se_i = se_eq[j][i]
            z_i  = β_i / se_i
            p_i  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_i)))
            lo   = β_i - zcrit * se_i
            hi   = β_i + zcrit * se_i
            push!(rows,
                  [first_row ? eqnames[j] : "",
                   cns[j][i],
                   Printf.@sprintf("%.7f", β_i),
                   Printf.@sprintf("%.7f", se_i),
                   Printf.@sprintf("%.2f", z_i),
                   Printf.@sprintf("%.3f", p_i),
                   Printf.@sprintf("%.7f", lo),
                   Printf.@sprintf("%.7f", hi)])
            first_row = false; nrows += 1
        end
        j < M && push!(hline_at, nrows)
    end
    _sug_print_table(coef_headers, rows,
                     [:l, :l, :r, :r, :r, :r, :r, :r];
                     top_hlines = hline_at)

    Printf.@printf("Endogenous: %s\n", join(sort(string.(collect(endog_set))), " "))
    Printf.@printf("Exogenous:  %s\n", join(string.(exog_set), " "))

    coefnames_full = String[]
    for j in 1:M, i in 1:ks[j]
        push!(coefnames_full, "[$(eqnames[j])]$(cns[j][i])")
    end
    return (; β = β_3sls, V = V_3sls, coefnames = coefnames_full,
              eqnames, cns, ks, offsets, N, Σ̂,
              β_eq, V_eq, se_eq, rmse, r2s, chi2s, pvals,
              endog = collect(endog_set), exog = exog_set)
end
