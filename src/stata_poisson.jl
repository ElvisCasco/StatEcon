# ============================================================================
# stata_poisson.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

"""
    stata_poisson(df, formula; vce=:robust, level=0.95, quiet=false,
                  layout=nothing, header_style=:poisson, cluster_var=nothing)

Stata-style `poisson y x…, vce(oim|robust|cluster)`. Fits a Poisson GLM with
log link (via `GLM.glm`) and prints Stata's coefficient table and header.

Keyword arguments
  - `vce`         : `:oim` (`GLM` observed-information vcov), `:robust`
                    (HC1 sandwich, n/(n−1) corrected — the default), or
                    `:cluster` (cluster-robust, requires `cluster_var`).
  - `cluster_var` : cluster identifier column; forces `vce = :cluster` and
                    uses Stata's `G/(G−1)` adjustment.
  - `level`       : confidence level for the reported CI (default 0.95).
  - `quiet`       : `true` suppresses the printed table (Stata `quietly`).
  - `layout`      : optional custom row layout (`:blank`, `label => key`,
                    `[hdr…, label] => key`, `label => :omitted`) for factor /
                    interaction displays. `nothing` = slopes then `_cons`.
  - `header_style`: `:poisson` (default) or `:ml` (Stata `ml maximize` block).

Returns a NamedTuple in **Stata display order** (slopes first, `_cons` last):

    model, df, formula,
    β, se, V, X, coefnames,          # display order; X's last col = ones
    Wald, Wald_p, ll, pseudo_r2, n, k,
    β_glm, V_glm, X_glm, coefnames_glm   # original GLM order

`β`/`V`/`X`/`coefnames` are the fields downstream margins / dydx / nlcom /
lincom helpers consume directly; `model`/`V_glm` (GLM order) drive helpers
that go through `StatsBase.coefnames(model)`.

Note: the constant `loggamma(y+1)` term of the Poisson log-likelihood is
handled internally by `GLM.loglikelihood`, so `ll` is the full Poisson log
(pseudo)likelihood — no term is dropped here.
"""
function stata_poisson(df, formula; vce::Symbol=:robust, level::Float64=0.95,
                       quiet::Bool=false,
                       layout::Union{Nothing,AbstractVector}=nothing,
                       header_style::Symbol=:poisson,
                       cluster_var=nothing)
    # Drop missing for variables in formula (and the cluster var, if any)
    needed = StatsModels.termvars(formula)
    cols   = cluster_var === nothing ? needed :
                                       vcat(needed, [Symbol(cluster_var)])
    dfc = DataFrames.dropmissing(df[:, cols])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end

    # Fit Poisson GLM
    m = GLM.glm(formula, dfc, Distributions.Poisson(), GLM.LogLink())
    β  = GLM.coef(m)
    cn = GLM.coefnames(m)
    n  = Int(StatsBase.nobs(m))
    yv = Float64.(GLM.response(m))
    μ  = GLM.predict(m)
    Xm = GLM.modelmatrix(m)
    k  = size(Xm, 2)

    # If `cluster_var` is given, force vce → :cluster regardless of caller's
    # choice (Stata's `vce(cluster <var>)` syntax), and capture cluster count.
    n_clusters = 0
    if cluster_var !== nothing
        vce = :cluster
        n_clusters = length(unique(dfc[!, Symbol(cluster_var)]))
    end

    # SE: robust sandwich (default) / cluster-robust (CR1) / OIM.
    # Stata's `vce(robust)` uses n/(n-1) finite-sample correction.
    # `vce(cluster G)` aggregates scores by cluster and uses G/(G-1).
    if vce == :robust
        H = Xm' * LinearAlgebra.Diagonal(μ) * Xm
        meat = Xm' * LinearAlgebra.Diagonal((yv .- μ).^2) * Xm
        Hinv = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H)))
        V = Hinv * meat * Hinv * (n / (n - 1))
    elseif vce == :cluster
        cluster_var === nothing &&
            error("vce = :cluster requires `cluster_var = :varname`")
        cl = dfc[!, Symbol(cluster_var)]
        u  = (yv .- μ) .* Xm                # n×k per-obs scores X_i·(y_i−μ_i)
        # Aggregate scores by cluster, then meat = Σ_g s_g s_g'.
        meat = zeros(k, k)
        for g in unique(cl)
            sg = vec(sum(u[cl .== g, :], dims = 1))
            meat .+= sg * sg'
        end
        H    = Xm' * LinearAlgebra.Diagonal(μ) * Xm
        Hinv = LinearAlgebra.inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H)))
        G    = n_clusters
        # Stata's `poisson, vce(cluster)` uses ONLY the G/(G−1) cluster
        # adjustment — no extra (n−1)/(n−k) factor (that one is OLS-style).
        V    = Hinv * meat * Hinv * (G / (G - 1))
    else
        V = Matrix(GLM.vcov(m))
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    p  = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    # Slopes (excluding intercept) — used for both Wald and LR
    intercept_idx = findfirst(==("(Intercept)"), cn)
    slope_idx = setdiff(1:length(β), [intercept_idx])
    Wald = NaN; Wald_p = NaN
    if !isempty(slope_idx)
        β_slope = β[slope_idx]
        V_slope = V[slope_idx, slope_idx]
        Wald = β_slope' * LinearAlgebra.inv(V_slope) * β_slope
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)), Wald)
    end

    # Log pseudolikelihood and Pseudo R² (McFadden's-like: 1 - LL/LL_null)
    ll  = GLM.loglikelihood(m)
    # Null model: intercept only
    m_null = GLM.glm(StatsModels.term(Symbol(formula.lhs)) ~ StatsModels.term(1),
                     dfc, Distributions.Poisson(), GLM.LogLink())
    ll_null = GLM.loglikelihood(m_null)
    pseudo_r2 = 1 - ll / ll_null

    # LR chi² on slopes — Stata's default `poisson` reports `LR chi2(k)`;
    # `vce(robust|cluster)` switches it to `Wald chi2(k)`.
    LR_chi2 = !isempty(slope_idx) ? 2 * (ll - ll_null) : NaN
    LR_p    = !isempty(slope_idx) ?
              1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)), LR_chi2) :
              NaN

    # Stata %9.0g formatter — sign-aware width cap.
    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
        end
        if 0 < abs(x) < 1
            s = replace(s, r"^(-?)0\." => s"\1.")
        end
        return lpad(s, w)
    end
    function commafmt(num)
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1
            push!(parts, s[max(1, i-2):i]); i -= 3
        end
        return (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    if !quiet
    println()
    if header_style == :ml
        nstr = commafmt(n)
        wstr = isfinite(Wald) ? Printf.@sprintf("%.2f", Wald) : ""
        pstr = isfinite(Wald) ? Printf.@sprintf("%.4f", Wald_p) : ""
        vw   = maximum(length, (nstr, wstr, pstr))
        rblock = 13 + 3 + vw
        lead   = max(0, 78 - rblock)
        ws     = " " ^ lead
        println(ws, rpad("Number of obs", 13), " = ", lpad(nstr, vw))
        if isfinite(Wald)
            println(ws, rpad("Wald chi2($(length(slope_idx)))", 13),
                    " = ", lpad(wstr, vw))
        end
        ll_label = (vce == :robust || vce == :cluster) ? "Log pseudolikelihood" : "Log likelihood"
        ll_str   = Printf.@sprintf("%s = %.3f", ll_label, ll)
        println(rpad(ll_str, lead), rpad("Prob > chi2", 13), " = ", lpad(pstr, vw))
    else
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Poisson regression", "Number of obs", commafmt(n))
        chi2_label = (vce == :robust || vce == :cluster) ? "Wald" : "LR"
        chi2_val   = (vce == :robust || vce == :cluster) ? Wald   : LR_chi2
        chi2_p     = (vce == :robust || vce == :cluster) ? Wald_p : LR_p
        if isfinite(chi2_val)
            Printf.@printf("%56s%-13s = %6.2f\n", "",
                           "$chi2_label chi2($(length(slope_idx)))", chi2_val)
            Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", chi2_p)
        end
        ll_label = (vce == :robust || vce == :cluster) ?
                   "Log pseudolikelihood" : "Log likelihood"
        ll_str = Printf.@sprintf("%s = %.5f", ll_label, ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad_h  = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad_h, r2_str)
    end
    println()

    if vce == :cluster
        cl_str = Printf.@sprintf("(Std. err. adjusted for %d clusters in %s)",
                                 n_clusters, cluster_var)
        println(lpad(cl_str, 78))
    end

    println("-"^78)
    if vce == :robust || vce == :cluster
        println("             |               Robust")
    end
    se_label = (vce == :robust || vce == :cluster) ? "std. err." : "Std. err."
    Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [95%% conf. interval]\n",
                   string(formula.lhs), se_label)
    println("-"^13, "+", "-"^64)

    _print_row(label, idx) = Printf.@printf(
        "%12s | %s  %s  %s   %s    %s  %s\n",
        label, g9(β[idx]; w=10), g9(se[idx]; w=9),
        Printf.@sprintf("%7.2f", z[idx]),
        Printf.@sprintf("%.3f", p[idx]),
        g9(ci_lo[idx]; w=9), g9(ci_hi[idx]; w=10))
    _lookup(key) = key isa Integer ? Int(key) :
        (let i = findfirst(==(string(key)), cn)
            i === nothing && error("'$(key)' not found in coefnames=$(cn)")
            i
        end)

    if layout === nothing
        for i in slope_idx
            _print_row(cn[i], i)
        end
        intercept_idx !== nothing && _print_row("_cons", intercept_idx)
    else
        for entry in layout
            if entry === :blank
                println("             |")
            elseif entry isa Pair
                lab, key = entry.first, entry.second
                if key === :omitted
                    Printf.@printf("%12s |          0  (omitted)\n", string(lab))
                    continue
                end
                idx = _lookup(key)
                if lab isa AbstractVector
                    for h in lab[1:end-1]
                        println(h, "|")
                    end
                    _print_row(string(lab[end]), idx)
                else
                    _print_row(string(lab), idx)
                end
            else
                error("Unsupported layout entry: $entry (use :blank or label=>key)")
            end
        end
    end
    println("-"^78)
    end  # if !quiet

    # Return values in *Stata's display order*: slopes first, then _cons.
    if intercept_idx === nothing
        ord = collect(slope_idx)
        cn_ord = cn[ord]
    else
        ord = vcat(collect(slope_idx), [intercept_idx])
        cn_ord = vcat(cn[slope_idx], "_cons")
    end
    β_ord  = β[ord]
    se_ord = se[ord]
    V_ord  = V[ord, ord]
    X_ord  = Xm[:, ord]   # last column = ones (the constant)

    return (; model=m, df=dfc, formula,
              β=β_ord, se=se_ord, V=V_ord, X=X_ord,
              coefnames=cn_ord,
              Wald, Wald_p, ll, pseudo_r2, n, k,
              β_glm=β, V_glm=V, X_glm=Xm, coefnames_glm=cn)
end
