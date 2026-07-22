# Seemingly-unrelated regression (SUR) — Stata's `sureg` command.
#
#   stata_sureg(df,
#               :y1 => [:x1a, :x1b],
#               :y2 => [:x2a, :x2b];
#               reps = 400, seed = 10101)
#
#   fit = stata_sureg_fit(df, :y1 => [...], :y2 => [...])
#   stata_test_sureg(fit, [:age, :age2])         # joint Wald test
#
# Silent fit (`stata_sureg_fit`) returns a NamedTuple; `stata_sureg` wraps it
# and prints Stata's summary block, the pooled coefficient table, the
# residual correlation matrix, and the Breusch-Pagan test of diagonality.
# Algorithm: equation-by-equation OLS gives Σ̂, one FGLS step with W = Σ̂⁻¹ ⊗ I
# recovers β̂_SUR and V̂_SUR. Pass `reps > 0, seed = ...` for a paired-bootstrap
# vcov (β̂ unchanged, V̂ replaced by the sample covariance of B bootstrapped
# SUR fits). `constraints` accepts a vector of "[eq]var = value" or
# "[eq1]var = [eq2]var" strings (short-form `var = 0` also works when
# unambiguous).

function stata_sureg_fit(df::DataFrames.AbstractDataFrame, eqs::Pair...;
                         reps::Int = 0, seed::Union{Int,Nothing} = nothing,
                         constraints::AbstractVector = String[])
    M = length(eqs)
    all_vars = Symbol[]
    for (y, xs) in eqs
        push!(all_vars, Symbol(y))
        xsv = xs isa Union{Symbol,AbstractString} ? [Symbol(xs)] :
              [Symbol(v) for v in xs]
        append!(all_vars, xsv)
    end
    d = DataFrames.dropmissing(df, unique(all_vars))
    N = DataFrames.nrow(d)

    Xs = Vector{Matrix{Float64}}(undef, M)
    Ys = Vector{Vector{Float64}}(undef, M)
    cns = Vector{Vector{String}}(undef, M)
    eqnames = Vector{String}(undef, M)
    for (j, (y, xs)) in enumerate(eqs)
        ys  = Symbol(y)
        xsv = xs isa Union{Symbol,AbstractString} ? [Symbol(xs)] :
              [Symbol(v) for v in xs]
        Xs[j] = hcat(ones(N), [Float64.(_sm_rawval.(d[!, v])) for v in xsv]...)
        Ys[j] = Float64.(_sm_rawval.(d[!, ys]))
        cns[j] = vcat("_cons", string.(xsv))
        eqnames[j] = string(ys)
    end
    ks = [size(X, 2) for X in Xs]
    kt = sum(ks)
    offsets = vcat(0, cumsum(ks))

    coefnames_full = String[]
    for j in 1:M, i in 1:ks[j]
        push!(coefnames_full, "[$(eqnames[j])]$(cns[j][i])")
    end

    # Parse "[eq]var = number" or "[eq1]var = [eq2]var" or short "var = 0"
    R_con = zeros(0, kt); r_con = Float64[]; con_labels = String[]
    if !isempty(constraints)
        function _resolve(token::AbstractString)
            tok = String(strip(token))
            num = tryparse(Float64, tok)
            num !== nothing && return (Int[], num)
            if startswith(tok, "[") && occursin(']', tok)
                k = findfirst(==(tok), coefnames_full)
                k === nothing && error("Coefficient not found: $tok")
                return ([k], 0.0)
            end
            idx = Int[]
            for (k, cn) in enumerate(coefnames_full)
                m = match(r"^\[(.+)\](.+)$", cn)
                (m === nothing || m.captures[2] != tok) && continue
                push!(idx, k)
            end
            isempty(idx) && error("Coefficient not found in any equation: $tok")
            return (idx, 0.0)
        end
        R_rows = Vector{Vector{Float64}}()
        for c in constraints
            cs = strip(string(c))
            occursin('=', cs) || error("Constraint must contain '=': got '$cs'")
            lhs_s, rhs_s = split(cs, '='; limit = 2)
            lhs_idx, lhs_c = _resolve(lhs_s)
            rhs_idx, rhs_c = _resolve(rhs_s)
            (length(lhs_idx) > 1 || length(rhs_idx) > 1) &&
                error("Ambiguous short name in '$cs' — use [eq]name")
            row = zeros(kt)
            for k in lhs_idx; row[k] += 1.0; end
            for k in rhs_idx; row[k] -= 1.0; end
            push!(R_rows, row); push!(r_con, rhs_c - lhs_c)
            push!(con_labels,
                  !isempty(lhs_idx) && !isempty(rhs_idx) ?
                      "$(coefnames_full[lhs_idx[1]]) = $(coefnames_full[rhs_idx[1]])" :
                  !isempty(lhs_idx) ?
                      "$(coefnames_full[lhs_idx[1]]) = $(rhs_c)" :
                      "$(coefnames_full[rhs_idx[1]]) = $(lhs_c)")
        end
        R_con = reduce(vcat, permutedims.(R_rows))
    end
    has_con = !isempty(con_labels)

    function _constrain(β_u, V_u)
        has_con || return (β_u, V_u)
        RVR = R_con * V_u * R_con'
        VRT = V_u * R_con'
        β_c = β_u .- VRT * (RVR \ (R_con * β_u .- r_con))
        V_c = V_u .- VRT * LinearAlgebra.inv(RVR) * R_con * V_u
        return (β_c, V_c)
    end

    # Step 1: OLS residuals → Σ̂
    β_ols = [Xs[j] \ Ys[j] for j in 1:M]
    e_ols = [Ys[j] - Xs[j] * β_ols[j] for j in 1:M]
    Σ̂ = [LinearAlgebra.dot(e_ols[i], e_ols[j]) / N for i in 1:M, j in 1:M]

    # Step 2: FGLS with W = Σ̂⁻¹ ⊗ I
    Σinv = LinearAlgebra.inv(Σ̂)
    XWX = zeros(kt, kt); XWY = zeros(kt)
    for i in 1:M, j in 1:M
        ri = (offsets[i]+1):offsets[i+1]
        rj = (offsets[j]+1):offsets[j+1]
        XWX[ri, rj] .= Σinv[i, j] .* (Xs[i]' * Xs[j])
        XWY[ri]    .+= Σinv[i, j] .* (Xs[i]' * Ys[j])
    end
    β_u_full = XWX \ XWY
    V_u_full = LinearAlgebra.inv(XWX)
    β_sur, V_sur = _constrain(β_u_full, V_u_full)

    β_eq = [β_sur[(offsets[j]+1):offsets[j+1]] for j in 1:M]
    V_eq = [V_sur[(offsets[j]+1):offsets[j+1], (offsets[j]+1):offsets[j+1]] for j in 1:M]
    se_eq = [sqrt.(max.(LinearAlgebra.diag(V), 0.0)) for V in V_eq]
    e_sur = [Ys[j] - Xs[j] * β_eq[j] for j in 1:M]

    rmse = [sqrt(LinearAlgebra.dot(e_sur[j], e_sur[j]) / N) for j in 1:M]
    r2s = Float64[]; chi2s = Float64[]; pvals = Float64[]
    for j in 1:M
        rss = LinearAlgebra.dot(e_sur[j], e_sur[j])
        tss = sum((Ys[j] .- Statistics.mean(Ys[j])) .^ 2)
        push!(r2s, 1 - rss / tss)
        slope = 2:ks[j]
        βs = β_eq[j][slope]; Vs = V_eq[j][slope, slope]
        χ2 = βs' * LinearAlgebra.pinv(Vs) * βs
        push!(chi2s, χ2)
        push!(pvals, 1 - Distributions.cdf(Distributions.Chisq(length(slope)), χ2))
    end

    # Optional bootstrap vcov
    if reps > 0
        seed !== nothing && Random.seed!(seed)
        β_boot = Matrix{Float64}(undef, reps, kt)
        for b in 1:reps
            idx = StatsBase.sample(1:N, N; replace = true)
            Xs_b = [X[idx, :] for X in Xs]
            Ys_b = [Y[idx] for Y in Ys]
            β_ols_b = [Xs_b[j] \ Ys_b[j] for j in 1:M]
            e_b = [Ys_b[j] - Xs_b[j] * β_ols_b[j] for j in 1:M]
            Σ_b = [LinearAlgebra.dot(e_b[i], e_b[j]) / N for i in 1:M, j in 1:M]
            Σinv_b = LinearAlgebra.inv(Σ_b)
            XWX_b = zeros(kt, kt); XWY_b = zeros(kt)
            for i in 1:M, j in 1:M
                ri = (offsets[i]+1):offsets[i+1]
                rj = (offsets[j]+1):offsets[j+1]
                XWX_b[ri, rj] .= Σinv_b[i, j] .* (Xs_b[i]' * Xs_b[j])
                XWY_b[ri]    .+= Σinv_b[i, j] .* (Xs_b[i]' * Ys_b[j])
            end
            β_ub = XWX_b \ XWY_b
            if has_con
                V_ub = LinearAlgebra.inv(XWX_b)
                β_ub, _ = _constrain(β_ub, V_ub)
            end
            β_boot[b, :] = β_ub
        end
        V_sur = Statistics.cov(β_boot)
        V_eq = [V_sur[(offsets[j]+1):offsets[j+1], (offsets[j]+1):offsets[j+1]] for j in 1:M]
        se_eq = [sqrt.(max.(LinearAlgebra.diag(V), 0.0)) for V in V_eq]
    end

    return (; β = β_sur, V = V_sur, coefnames = coefnames_full,
              eqnames, cns, ks, offsets, N, Σ̂,
              β_eq, V_eq, se_eq, rmse, r2s, chi2s, pvals, reps,
              constraints = con_labels)
end

# ---------- Printf table helpers (shared with survey/cluster files) --------

_sug_fmt(x, w, d) = @sprintf("%*.*f", w, d, x)

# Print a right-aligned table given column headers, formatters, and rows.
function _sug_print_table(headers::Vector{String}, rows::Vector{<:Vector},
                          aligns::Vector{Symbol};
                          top_hlines::Vector{Int} = Int[])
    ncols = length(headers)
    # Column widths: max of header/rows
    widths = [max(length(headers[c]),
                  maximum(length(String(r[c])) for r in rows; init = 0))
              for c in 1:ncols]
    padcell(s, w, a) = a === :l ? rpad(s, w) : lpad(s, w)
    rule = "-" ^ (sum(widths) + 3 * (ncols - 1))
    println(rule)
    println(join([padcell(headers[c], widths[c], aligns[c]) for c in 1:ncols], "   "))
    println(rule)
    for (i, r) in enumerate(rows)
        println(join([padcell(String(r[c]), widths[c], aligns[c]) for c in 1:ncols], "   "))
        i in top_hlines && println(rule)
    end
    println(rule)
end

_sug_g(x::Real; d = 4) = isnan(x) ? "." : (isinteger(x) && abs(x) < 1e6 ? string(Int(x)) : @sprintf("%.*f", d, x))

# --------------------------------------------------------------------------

function stata_sureg(df::DataFrames.AbstractDataFrame, eqs::Pair...;
                     level::Float64 = 0.95, reps::Int = 0,
                     seed::Union{Int,Nothing} = nothing,
                     constraints::AbstractVector = String[])
    fit = stata_sureg_fit(df, eqs...; reps = reps, seed = seed,
                          constraints = constraints)
    if !isempty(fit.constraints)
        println("Constraints imposed:")
        for (i, c) in enumerate(fit.constraints)
            @printf(" ( %d)  %s\n", i, c)
        end
        println()
    end
    M = length(fit.eqnames)
    eqnames = fit.eqnames; cns = fit.cns; ks = fit.ks; N = fit.N
    β_eq = fit.β_eq; se_eq = fit.se_eq
    rmse = fit.rmse; r2s = fit.r2s; chi2s = fit.chi2s; pvals = fit.pvals
    Σ̂ = fit.Σ̂

    println("Seemingly unrelated regression\n")
    # Summary block
    sum_headers = ["Equation", "Obs", "Params", "RMSE", "R-squared", "chi2", "P>chi2"]
    sum_rows = Vector{Vector{String}}()
    for j in 1:M
        push!(sum_rows,
              [eqnames[j], string(N), string(ks[j] - 1),
               @sprintf("%.6f", rmse[j]),
               @sprintf("%.4f", r2s[j]),
               @sprintf("%.2f", chi2s[j]),
               @sprintf("%.4f", pvals[j])])
    end
    _sug_print_table(sum_headers, sum_rows,
                     [:l, :r, :r, :r, :r, :r, :r])

    # Coefficient table (slopes first, intercept last)
    zcrit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    lvl = round(Int, 100 * level)
    coef_headers = reps > 0 ?
        ["Equation", "Variable", "Observed coef.", "Bootstrap SE",
         "z", "P>|z|", "[$(lvl)% CI low]", "[$(lvl)% CI high]"] :
        ["Equation", "Variable", "Coefficient", "Std. err.",
         "z", "P>|z|", "[$(lvl)% CI low]", "[$(lvl)% CI high]"]
    rows = Vector{Vector{String}}()
    hline_at = Int[]
    nrows = 0
    for j in 1:M
        order = vcat(collect(2:ks[j]), 1)
        first_row = true
        for i in order
            β_i = β_eq[j][i]; se_i = se_eq[j][i]
            z_i = β_i / se_i
            p_i = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z_i)))
            push!(rows,
                  [first_row ? eqnames[j] : "",
                   cns[j][i],
                   @sprintf("%.7f", β_i),
                   @sprintf("%.7f", se_i),
                   @sprintf("%.2f", z_i),
                   @sprintf("%.3f", p_i),
                   @sprintf("%.7f", β_i - zcrit * se_i),
                   @sprintf("%.7f", β_i + zcrit * se_i)])
            first_row = false; nrows += 1
        end
        j < M && push!(hline_at, nrows)
    end
    println()
    _sug_print_table(coef_headers, rows,
                     [:l, :l, :r, :r, :r, :r, :r, :r]; top_hlines = hline_at)
    reps > 0 && @printf("  (bootstrap replications = %d)\n", reps)

    # Residual correlation matrix (from OLS-step Σ̂)
    Rmat = [Σ̂[i, j] / sqrt(Σ̂[i, i] * Σ̂[j, j]) for i in 1:M, j in 1:M]
    println("\nCorrelation matrix of residuals:\n")
    corr_headers = vcat("", eqnames)
    corr_rows = [vcat(eqnames[i],
                      [j <= i ? @sprintf("%.4f", Rmat[i, j]) : ""
                       for j in 1:M]) for i in 1:M]
    _sug_print_table(corr_headers, corr_rows, vcat(:l, fill(:r, M)))

    # Breusch–Pagan test of independence
    bp = 0.0
    for i in 1:M-1, j in (i+1):M
        bp += Rmat[i, j]^2
    end
    bp *= N
    df_bp = M * (M - 1) ÷ 2
    p_bp = 1 - Distributions.cdf(Distributions.Chisq(df_bp), bp)
    @printf("\nBreusch-Pagan test of independence: chi2(%d) = %.3f, Pr = %.4f\n",
            df_bp, bp, p_bp)
    return fit
end

function stata_test_sureg(fit, vars::AbstractVector;
                          df_F::Union{Integer,Nothing} = nothing)
    coefnames = fit.coefnames
    K = length(coefnames)

    function _resolve(token::AbstractString)
        tok = String(strip(token))
        num = tryparse(Float64, tok)
        num !== nothing && return (Int[], num)
        if startswith(tok, "[") && occursin(']', tok)
            k = findfirst(==(tok), coefnames)
            k === nothing && error("Coefficient not found: $tok")
            return ([k], 0.0)
        end
        idx = Int[]
        for (k, cn) in enumerate(coefnames)
            m = match(r"^\[(.+)\](.+)$", cn)
            (m === nothing || m.captures[2] != tok) && continue
            push!(idx, k)
        end
        isempty(idx) && error("Coefficient not found in any equation: $tok")
        return (idx, 0.0)
    end

    R_rows = Vector{Vector{Float64}}()
    r_vals = Float64[]
    labels = String[]
    for v in vars
        vs = strip(string(v))
        if occursin('=', vs)
            lhs_s, rhs_s = split(vs, '='; limit = 2)
            lhs_idx, lhs_c = _resolve(lhs_s)
            rhs_idx, rhs_c = _resolve(rhs_s)
            length(lhs_idx) > 1 &&
                error("Ambiguous short name on lhs of '$vs' — use [eq]name")
            length(rhs_idx) > 1 &&
                error("Ambiguous short name on rhs of '$vs' — use [eq]name")
            row = zeros(K)
            for k in lhs_idx; row[k] += 1.0; end
            for k in rhs_idx; row[k] -= 1.0; end
            push!(R_rows, row); push!(r_vals, rhs_c - lhs_c)
            push!(labels,
                  !isempty(lhs_idx) && !isempty(rhs_idx) ?
                      "$(coefnames[lhs_idx[1]]) - $(coefnames[rhs_idx[1]]) = 0" :
                  !isempty(lhs_idx) ?
                      "$(coefnames[lhs_idx[1]]) = $(rhs_c)" :
                      "$(coefnames[rhs_idx[1]]) = $(lhs_c)")
        else
            idx_all, _ = _resolve(vs)
            for k in idx_all
                row = zeros(K); row[k] = 1.0
                push!(R_rows, row); push!(r_vals, 0.0)
                push!(labels, "$(coefnames[k]) = 0")
            end
        end
    end
    isempty(R_rows) && error("No restrictions parsed from: $(vars)")

    R = reduce(vcat, permutedims.(R_rows))
    r = r_vals
    d = R * fit.β .- r
    W = d' * LinearAlgebra.inv(R * fit.V * R') * d
    q = length(r)

    for (i, lbl) in enumerate(labels)
        @printf(" ( %d)  %s\n", i, lbl)
    end
    println()

    if df_F === nothing
        p_chi = 1 - Distributions.cdf(Distributions.Chisq(q), W)
        @printf("           chi2(%3d) = %7.2f\n", q, W)
        @printf("         Prob > chi2 = %7.4f\n", p_chi)
        return (; chi2 = W, df = q, p = p_chi, restrictions = labels, R, r)
    else
        Fs = W / q
        p_F = 1 - Distributions.cdf(Distributions.FDist(q, df_F), Fs)
        @printf("       F(%3d, %5d) = %7.2f\n", q, df_F, Fs)
        @printf("            Prob > F = %7.4f\n", p_F)
        return (; F = Fs, df1 = q, df2 = df_F, p = p_F,
                  chi2 = W, restrictions = labels, R, r)
    end
end
