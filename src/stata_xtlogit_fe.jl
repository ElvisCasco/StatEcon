# (deps provided by the StatEcon module)
import Optim

"""
    stata_xtlogit_fe(df; depvar, regs, idvar, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xtlogit <depvar> <regs>, fe nolog` — Chamberlain (1980)
conditional fixed-effects logit. The fixed effects α_i are eliminated
by conditioning on the panel sufficient statistic Σ_t y_it. The
conditional log-likelihood is

    ℓ_i(β) = Σ_{t∈S_obs_i} x_it'β  −  log Σ_{S ∈ subsets(T_i, c_i)}
                                          exp(Σ_{t∈S} x_it'β)

where `c_i = Σ_t y_it` and the outer sum runs over all `C(T_i, c_i)`
binary sequences of length `T_i` summing to `c_i`. Panels with all-0
or all-1 outcomes contribute no information and are dropped.

Time-invariant regressors (no within-panel variation) are reported as
`0  (omitted)` and do not enter estimation. The LR test of overall
significance uses `LR = 2·(LL − LL_null)` where the null fit has zero
parameters and `LL_null = −Σ_i log C(T_i, c_i)`.

Output mirrors Stata's `xtlogit, fe` block: header (Number of obs /
groups / Obs per group / LR chi² / Log likelihood / Prob > chi²) and
the coefficient table with `(omitted)` rows for time-invariant
regressors.

Returns `(; β, V, se, coefnames_kept, omitted, ll, ll_null, LR, LR_p,
n, n_panels, T_min, T_max, T_avg)`.
"""
function stata_xtlogit_fe(df; depvar::Symbol,
                          regs::AbstractVector{Symbol},
                          idvar::Symbol,
                          level::Float64 = 0.95,
                          quiet::Bool = false)
    needed = unique(vcat(depvar, idvar, regs))
    d = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = d[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            d[!, c] = Float64.(col)
        end
    end
    keep = trues(DataFrames.nrow(d))
    for c in needed
        col = d[!, c]
        eltype(col) <: Real || continue
        for i in eachindex(col); keep[i] &= isfinite(col[i]); end
    end
    d = d[keep, :]
    d = DataFrames.sort(d, [idvar])

    panels_all = DataFrames.groupby(d, idvar)
    pd_all = [(y = Float64.(g[!, depvar]),
               X = hcat([Float64.(g[!, v]) for v in regs]...))
              for g in panels_all]

    # Drop uninformative panels (all 0 or all 1).
    pd = filter(p -> 0 < sum(p.y) < length(p.y), pd_all)
    n_panels = length(pd)
    n_obs    = sum(length(p.y) for p in pd)

    # Detect time-invariant regressors: no within-panel variation across
    # all kept panels. NOTE: nested `for` blocks (not the Cartesian
    # `for j..., p..., end` form) so the inner `break` only exits the
    # panel scan for the current j.
    n_regs = length(regs)
    is_tv  = falses(n_regs)
    for j in 1:n_regs
        for p in pd
            col = view(p.X, :, j)
            # A regressor is "time-varying" if there exists a panel with
            # within-panel variation. Stata's rule.
            if length(unique(col)) > 1
                is_tv[j] = true
                break
            end
        end
    end
    tv_idx      = findall(is_tv)
    omitted_idx = findall(.!is_tv)
    cnames_kept = string.(regs[tv_idx])
    omitted     = string.(regs[omitted_idx])
    k = length(tv_idx)

    # Reduce X to time-varying columns only.
    pd_tv = [(y = p.y, X = p.X[:, tv_idx]) for p in pd]

    # Pre-compute combinations per panel (subsets of size c_i).
    function _combinations(n_t::Int, k_c::Int)
        result = Vector{Vector{Int}}()
        cur = Int[]
        function helper(start::Int)
            if length(cur) == k_c
                push!(result, copy(cur))
                return
            end
            for i in start:n_t
                push!(cur, i); helper(i + 1); pop!(cur)
            end
        end
        helper(1)
        return result
    end
    panel_meta = [(; p_idx = i,
                     y     = pd_tv[i].y,
                     X     = pd_tv[i].X,
                     T_i   = length(pd_tv[i].y),
                     c_i   = Int(round(sum(pd_tv[i].y))),
                     obs_idx = findall(==(1), pd_tv[i].y),
                     combos  = _combinations(length(pd_tv[i].y),
                                             Int(round(sum(pd_tv[i].y)))))
                  for i in 1:n_panels]

    # ll_null: with zero regressors, conditional likelihood is uniform → 1/|C|.
    ll_null = -sum(log(Float64(length(pm.combos))) for pm in panel_meta)

    # Negative conditional log-likelihood.
    function negll(β)
        ll = zero(eltype(β))
        for pm in panel_meta
            X = pm.X; obs = pm.obs_idx
            num = zero(eltype(β))
            for t in obs
                num += LinearAlgebra.dot(view(X, t, :), β)
            end
            log_terms = Vector{eltype(β)}(undef, length(pm.combos))
            for (q, S) in enumerate(pm.combos)
                acc = zero(eltype(β))
                for t in S
                    acc += LinearAlgebra.dot(view(X, t, :), β)
                end
                log_terms[q] = acc
            end
            mlt = maximum(log_terms)
            ll += num - (mlt + log(sum(exp.(log_terms .- mlt))))
        end
        return -ll
    end

    β0 = zeros(k)
    if k > 0
        res = _c18_optimize(negll, β0, Optim.LBFGS(),
                             Optim.Options(g_tol = 1e-8, iterations = 2000))
        β  = Optim.minimizer(res)
        ll = -negll(β)
    else
        β = β0
        ll = ll_null
    end

    # Finite-difference Hessian for SEs (only over time-varying regs).
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
    V  = k > 0 ? LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, β))) :
                 zeros(0, 0)
    se = k > 0 ? sqrt.(max.(LinearAlgebra.diag(V), 0.0)) : Float64[]
    z  = k > 0 ? β ./ se : Float64[]
    pv = k > 0 ?
         2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z))) :
         Float64[]
    crit  = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = k > 0 ? β .- crit .* se : Float64[]
    ci_hi = k > 0 ? β .+ crit .* se : Float64[]

    LR   = max(2 * (ll - ll_null), 0.0)
    LR_p = k > 0 ?
           1 - Distributions.cdf(Distributions.Chisq(k), LR) :
           NaN

    T_per = [pm.T_i for pm in panel_meta]
    T_min = minimum(T_per); T_max = maximum(T_per)
    T_avg = Statistics.mean(T_per)

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        sig_use = sig
        s = Printf.@sprintf("%.*g", sig_use, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && sig_use > 1
            sig_use -= 1
            s = Printf.@sprintf("%.*g", sig_use, x)
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
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Conditional fixed-effects logistic regression",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        println()
        Printf.@printf("%-53s%s\n", "", "Obs per group:")
        Printf.@printf("%-53s%18s = %6d\n", "", "min", T_min)
        Printf.@printf("%-53s%18s = %6.1f\n", "", "avg", T_avg)
        Printf.@printf("%-53s%18s = %6d\n", "", "max", T_max)
        println()
        Printf.@printf("%-53s%-17s= %6s\n", "",
                       "LR chi2($k)", Printf.@sprintf("%.2f", LR))
        ll_str  = Printf.@sprintf("Log likelihood = %.4f", ll)
        right   = Printf.@sprintf("%-17s= %6s", "Prob > chi2",
                                  Printf.@sprintf("%.4f", LR_p))
        pad_h   = max(0, 78 - length(ll_str) - length(right))
        println(ll_str, " "^pad_h, right)
        println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100*level)
        println("-"^13, "+", "-"^64)
        # Print rows in formula order so omitted variables sit alongside kept ones.
        for j in 1:length(regs)
            name = string(regs[j])
            if is_tv[j]
                ki = findfirst(==(j), tv_idx)
                Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                               name, g9(β[ki]; w=10), g9(se[ki]; w=9),
                               Printf.@sprintf("%7.2f", z[ki]),
                               Printf.@sprintf("%.3f", pv[ki]),
                               g9(ci_lo[ki]; w=9), g9(ci_hi[ki]; w=10))
            else
                Printf.@printf("%12s |          0  (omitted)\n", name)
            end
        end
        println("-"^78)
    end

    return (; β, V, se, coefnames_kept = cnames_kept, omitted,
              ll, ll_null, LR, LR_p,
              n = n_obs, n_panels, T_min, T_max, T_avg)
end

