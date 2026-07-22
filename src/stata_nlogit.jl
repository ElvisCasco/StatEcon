import Optim

"""
    stata_nlogitgen(df, alt_col; gen, nests) -> Dict{Int,String}

Stata `nlogitgen <gen> = <alt_col>(<nest>: <alts> | <alts>, …)`. Adds
an integer-coded nest column to `df` and returns the label dict
`Int ⇒ nest-name`. `nests` is an ordered iterable of
`nest_name => [alt_names…]` pairs — encoding starts at 1 and follows
that order, matching Stata's left-to-right read of the `nlogitgen`
call. Prints Stata's `new variable … generated with N groups` notice
plus the `label list <gen>` block.
"""
function stata_nlogitgen(df::DataFrames.AbstractDataFrame, alt_col::Symbol;
                         gen::Symbol,
                         nests::AbstractVector)
    label_of   = Dict{String,Int}()
    nest_label = Dict{Int,String}()
    for i in eachindex(nests)
        pr = nests[i]
        # Accept Pair / Tuple / Vector — anything with two positional
        # parts. Destructuring inside a for-loop over Vector{Pair} is
        # surprisingly brittle (the iteration protocol on Pair yields
        # both halves separately, so `nest_name, alts = pr` consumes
        # one element per side and leaves `alts` as a single string).
        nest_name = pr isa Pair ? pr.first  : pr[1]
        alts      = pr isa Pair ? pr.second : pr[2]
        nest_label[i] = String(nest_name)
        for a in alts
            label_of[String(a)] = i
        end
    end
    function _lookup(v)
        ismissing(v) && return missing
        s = string(v)
        haskey(label_of, s) ||
            error("$(alt_col) value `$(s)` not in any nest")
        return label_of[s]
    end
    df[!, gen] = [_lookup(v) for v in df[!, alt_col]]
    Printf.@printf("new variable %s is generated with %d groups\n",
                   string(gen), length(nests))
    stata_label_list(nest_label; name = string(gen))
    return nest_label
end

"""
    stata_nlogittree(df, alt_col, nest_col; choice, nests=nothing)

Stata `nlogittree <alt_col> <nest_col>, choice(<choice>)`. Prints the
nest tree with N (rows per nest / per alt) and k (times chosen, i.e.
`<choice> == 1`) columns. If `nests` is given (same vector-of-pairs
form as `stata_nlogitgen`), the within-nest alternative order follows
that spec; otherwise alternatives appear in the order they first
appear in the data within each nest.
"""
function stata_nlogittree(df::DataFrames.AbstractDataFrame,
                          alt_col::Symbol,
                          nest_col::Symbol;
                          choice::Symbol,
                          nests::Union{Nothing,AbstractVector} = nothing)
    altv = string.(df[!, alt_col])
    nestv = df[!, nest_col]
    ch    = [Float64(_c15_raw(v)) for v in df[!, choice]]

    nest_keys = unique(nestv)
    # Build per-nest alt order (Stata respects the `nlogitgen` order).
    nest_alts = Dict{Any, Vector{String}}()
    if nests === nothing
        for k in nest_keys
            mask = nestv .== k
            nest_alts[k] = unique(altv[mask])
        end
    else
        for (i, pr) in enumerate(nests)
            alts_in = pr isa Pair ? pr.second : pr[2]
            nest_alts[i] = String.(alts_in)
        end
    end
    # Resolve nest display names: prefer string labels if nestv is a
    # LabeledVector or if `nests` was provided (use its key as label).
    nest_name = Dict{Any,String}()
    if nests === nothing
        for k in nest_keys
            v = nestv[findfirst(==(k), nestv)]
            nest_name[k] =
                hasproperty(v, :labels) && hasproperty(v, :value) &&
                haskey(v.labels, v.value) ? v.labels[v.value] : string(k)
        end
    else
        for (i, pr) in enumerate(nests)
            nm = pr isa Pair ? pr.first : pr[1]
            nest_name[i] = String(nm)
        end
    end

    println()
    println("tree structure specified for the nested logit model")
    println()
    # Column widths
    w_nest = max(length(string(nest_col)),
                 maximum(length, values(nest_name); init = 0))
    w_alt  = max(length(string(alt_col)),
                 maximum((length(a) for v in values(nest_alts) for a in v);
                         init = 0))
    # Header
    Printf.@printf("  %-*s  %5s   %-*s  %5s  %4s\n",
                   w_nest, string(nest_col), "N",
                   w_alt,  string(alt_col),  "N", "k")
    total_cases = 0
    for k in nest_keys
        nm = nest_name[k]
        nest_mask = nestv .== k
        n_nest = sum(nest_mask)
        first_in_nest = true
        for a in nest_alts[k]
            mask = nest_mask .& (altv .== a)
            n_alt = sum(mask)
            n_chosen = Int(sum(ch[mask]))
            if first_in_nest
                Printf.@printf("  %-*s  %5d   %-*s  %5d  %4d\n",
                               w_nest, nm, n_nest, w_alt, a, n_alt, n_chosen)
                first_in_nest = false
            else
                Printf.@printf("  %-*s  %5s   %-*s  %5d  %4d\n",
                               w_nest, "", "", w_alt, a, n_alt, n_chosen)
            end
        end
        total_cases += sum(ch[nest_mask])
    end
    println()
    Printf.@printf("total cases: %d\n", Int(total_cases))
    return nothing
end

"""
    stata_nlogit(df_long; case_var, alt_var, nest_col, depvar,
                 alt_specific=Symbol[], case_vars=Symbol[],
                 basealt=nothing, nests=nothing,
                 level=0.95, quiet=false)
        -> NamedTuple

Stata `nlogit <depvar> <alt_specific> || <nest>: , base(<basenest>) || <alt_var>: <case_vars>, case(<case_var>) notree nolog`.

Two-level nested logit on long-form data. For each (case i, alt j in
nest k(j)), utility

    V_ij = Σ_c β_c · z_ijc + α_j + Σ_v γ_jv · x_iv

with αⱼ ≡ 0 and γ_jv ≡ 0 for the base alternative (Stata picks the
first alternative alphabetically — pass `basealt` to override).
Dissimilarity parameters `τ_k` are estimated per nest.

Choice probability for chosen alternative j* in nest k* (Stata's
*RUM-consistent* nested logit form, the textbook default — within-
nest probability is the softmax of V/τ_k):

    P(j*|i) = exp(V_{ij*} / τ_{k*}) / D_{k*}        (within-nest term)
            × D_{k*}^{τ_{k*}} / Σ_m D_m^{τ_m}       (between-nest term)

where D_k = Σ_{l ∈ nest k} exp(V_il / τ_k).

Optimisation by LBFGS with autodiff gradient. τ is parameterised as
exp(η) (positive, *unbounded above* — matches Stata's "RUM-consistent
nested logit" output, which reports τ that can exceed 1 with the
corresponding RUM constraint violation; LL is the unconstrained MLE).
OIM vcov via FD Hessian, with pseudo-inverse fallback when the
Hessian is singular at the converged point.

Returns `(; β_alt, α, γ, τ, θ̂, V, se, ll, n_obs, n_cases, J, alts,
nonbase, nest_names, basealt, nests)`.
"""
function stata_nlogit(df_long::DataFrames.AbstractDataFrame;
                      case_var::Symbol,
                      alt_var::Symbol,
                      nest_col::Symbol,
                      depvar::Symbol,
                      alt_specific::AbstractVector{Symbol} = Symbol[],
                      case_vars::AbstractVector{Symbol}    = Symbol[],
                      basealt = nothing,
                      nests::Union{Nothing,AbstractVector} = nothing,
                      level::Float64 = 0.95,
                      quiet::Bool = false)
    # Alternatives (Stata order).
    alts = String[]
    for v in df_long[!, alt_var]
        s = string(v); s ∈ alts || push!(alts, s)
    end
    J = length(alts)
    basealt === nothing && (basealt = sort(alts)[1])   # alphabetical default
    base_s = string(basealt)
    nonbase = filter(!=(base_s), alts)
    Jm1 = J - 1

    # Nest spec — either parsed from `nests` Pair vector or read from
    # the integer-coded `nest_col` (1..K).
    nest_of_alt = Dict{String,Int}()
    nest_names  = String[]
    if nests === nothing
        nest_keys = sort(unique(df_long[!, nest_col]))
        for (k, nk) in enumerate(nest_keys)
            push!(nest_names, string(nk))
            mask = df_long[!, nest_col] .== nk
            for s in unique(string.(df_long[mask, alt_var]))
                nest_of_alt[s] = k
            end
        end
    else
        for (k, pr) in enumerate(nests)
            nm = pr isa Pair ? pr.first  : pr[1]
            aa = pr isa Pair ? pr.second : pr[2]
            push!(nest_names, String(nm))
            for s in aa; nest_of_alt[String(s)] = k; end
        end
    end
    K = length(nest_names)

    # Per-row precomputed indices/data.
    as_float(col) = [Float64(_c15_raw(v)) for v in col]
    y    = as_float(df_long[!, depvar])
    cs   = df_long[!, case_var]
    altv = [string(v) for v in df_long[!, alt_var]]
    Z    = [as_float(df_long[!, v]) for v in alt_specific]
    X    = [as_float(df_long[!, v]) for v in case_vars]
    N    = length(y)

    # Group rows by case.
    case_ids = unique(cs)
    G = length(case_ids)
    grp_rows = Dict{eltype(case_ids), Vector{Int}}()
    for i in 1:N; push!(get!(() -> Int[], grp_rows, cs[i]), i); end

    nb_idx = Dict(s => k for (k, s) in enumerate(nonbase))   # 1..(J-1)
    nest_idx = [nest_of_alt[a] for a in altv]                # per-row

    # Parameter layout: [β_alt (p_alt), α_nonbase (Jm1),
    #                    γ_v1_nonbase, γ_v2_nonbase, …, η_τ (K)]
    p_alt  = length(alt_specific)
    p_csv  = length(case_vars)
    nparam = p_alt + Jm1 + Jm1 * p_csv + K
    function unpack(θ)
        β = view(θ, 1:p_alt)
        α_nb = view(θ, p_alt+1 : p_alt+Jm1)
        γ_nb = [view(θ, p_alt + Jm1 + (v-1)*Jm1 + 1 : p_alt + Jm1 + v*Jm1)
                for v in 1:p_csv]
        η_τ  = view(θ, nparam - K + 1 : nparam)
        # Map η → τ = exp(η) (positive, unbounded above). Stata's
        # "RUM-consistent nlogit" reports unconstrained τ even when it
        # exceeds 1 (with the corresponding RUM-violation flag); this
        # parametrisation matches that behaviour and the textbook fit.
        τ = [exp(η_τ[k]) for k in 1:K]
        return β, α_nb, γ_nb, τ
    end

    function utility_at(θ, rowidx)
        β, α_nb, γ_nb, _ = unpack(θ)
        u = zero(eltype(θ))
        for c in 1:p_alt; u += β[c] * Z[c][rowidx]; end
        a = altv[rowidx]
        if a != base_s
            k = nb_idx[a]
            u += α_nb[k]
            for v in 1:p_csv; u += γ_nb[v][k] * X[v][rowidx]; end
        end
        return u
    end

    function negll(θ)
        T = eltype(θ)
        _, _, _, τ = unpack(θ)
        ll = zero(T)
        for (_, ridx) in grp_rows
            # RUM-consistent (UMNL) log-sum-exp: V scaled by τ_k
            # *inside* the nest. D_k = Σ exp(V_il/τ_k). Buffers must
            # carry eltype(θ) so ForwardDiff Duals can be stored — see
            # feedback-forwarddiff-buffer-eltype.
            sums  = zeros(T, K)
            maxes = fill(T(-Inf), K)
            for i in ridx
                k = nest_idx[i]
                v_over_τ = utility_at(θ, i) / τ[k]
                if v_over_τ > maxes[k]
                    sums[k] *= exp(maxes[k] - v_over_τ)
                    maxes[k] = v_over_τ
                end
                sums[k] += exp(v_over_τ - maxes[k])
            end
            # logD_k = log Σ exp(V/τ_k) = maxes_k + log(sums_k)
            logD = [sums[k] > 0 ? maxes[k] + log(sums[k]) : T(-Inf)
                    for k in 1:K]
            # Top-level denominator: Σ_m D_m^{τ_m} = Σ_m exp(τ_m logD_m)
            top = [τ[k] * logD[k] for k in 1:K]
            top_max = maximum(top)
            top_sum = sum(exp.(top .- top_max))
            log_top = top_max + log(top_sum)
            for i in ridx
                if y[i] > 0
                    k = nest_idx[i]
                    v = utility_at(θ, i)
                    # log P(j*|i) = V/τ_k + (τ_k - 1) logD_k - log_top
                    ll += v / τ[k] + (τ[k] - 1) * logD[k] - log_top
                end
            end
        end
        return -ll
    end

    # Warm start: zeros for β / α / γ; η_τ = 0 → τ = 0.5.
    θ0 = zeros(nparam)
    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 4000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    ll = -Optim.minimum(res)

    # OIM vcov via FD Hessian, then delta-method back from η to τ.
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
                    (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h_[i]*h_[j])
            end
        end
        return H
    end
    # The logistic re-parametrisation of τ has gradient τ(1-τ), which
    # vanishes at the boundary. When τ_k hits 0 or 1 the η_k column of
    # the Hessian goes singular — `inv` raises `SingularException`. Use
    # `pinv` (Moore–Penrose pseudo-inverse) so the SE matrix is still
    # well-defined; the corresponding SE for a boundary τ is reported
    # as `NaN`-equivalent (zero variance row → zero SE), matching what
    # Stata shows when a dissimilarity parameter is on the boundary.
    H_sym = LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂))
    V_raw = try
        LinearAlgebra.inv(H_sym)
    catch err
        err isa LinearAlgebra.SingularException ?
            LinearAlgebra.pinv(Matrix(H_sym)) : rethrow(err)
    end
    # Delta-method transform: τ = exp(η) ⇒ dτ/dη = τ. Build Jacobian
    # for the full parameter vector (identity rows for β/α/γ, derivative
    # for the τ block).
    β̂, α̂_nb, γ̂_nb, τ̂  = unpack(θ̂)
    Jmat = Matrix{Float64}(LinearAlgebra.I, nparam, nparam)
    for k in 1:K
        i = nparam - K + k
        Jmat[i, i] = τ̂[k]
    end
    V = Jmat * V_raw * Jmat'
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    # Unpack results into Dicts.
    α = Dict{String,Float64}(base_s => 0.0)
    γ = Dict{String,Vector{Float64}}(base_s => zeros(p_csv))
    for (k, s) in enumerate(nonbase)
        α[s] = α̂_nb[k]
        γ[s] = [γ̂_nb[v][k] for v in 1:p_csv]
    end
    τ = Dict{String,Float64}(nest_names[k] => τ̂[k] for k in 1:K)

    if !quiet
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
        crit = Distributions.quantile(Distributions.Normal(),
                                      1 - (1 - level) / 2)
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Nested logit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("Case variable: %s%*s%-13s = %6s\n",
                       string(case_var),
                       max(0, 56 - 15 - length(string(case_var))), "",
                       "Number of cases", commafmt(G))
        println()
        Printf.@printf("Alternative variable: %s\n", string(alt_var))
        println()
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        println(ll_str); println()
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)

        function _print(label, b, s)
            z = b / s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit * s; hi = b + crit * s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w = 10), g9(s; w = 9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w = 9), g9(hi; w = 10))
        end
        # Alt-specific block
        if p_alt > 0
            Printf.@printf("%-12s |\n", string(alt_var))
            for c in 1:p_alt
                _print(string(alt_specific[c]), θ̂[c], se[c])
            end
        end
        # Per non-base alternative: γ rows + _cons
        for (k, s_alt) in enumerate(nonbase)
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", s_alt)
            for v in 1:p_csv
                idx = p_alt + Jm1 + (v - 1) * Jm1 + k
                _print(string(case_vars[v]), θ̂[idx], se[idx])
            end
            _print("_cons", θ̂[p_alt + k], se[p_alt + k])
        end
        # Base alternative row
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |  (base alternative)\n", base_s)
        # Dissimilarity parameters
        println("-"^13, "+", "-"^64)
        println("dissimilarity parameters |")
        for k in 1:K
            idx = nparam - K + k
            _print(nest_names[k] * " /tau", τ̂[k], se[idx])
        end
        println("-"^78)
    end

    return (; β_alt = β̂, α, γ, τ, θ̂, V, se, ll,
              n_obs = N, n_cases = G, J, K, alts, nonbase, nest_names,
              basealt = base_s, nests, nparam, nest_idx,
              alt_specific = collect(alt_specific),
              case_vars    = collect(case_vars),
              case_var_name = case_var, alt_var_name = alt_var,
              depvar_name = depvar, nest_col_name = nest_col,
              nest_of_alt)
end

"""
    _c15_nlogit_probs(res, df_long; case_var, alt_var) -> NamedTuple

Stata `predict plevel1 plevel2, pr` after `nlogit` (RUM-consistent
form). Stata numbers the tree from the top down — level 1 is the
nest, level 2 is the alternative. For each `(case, alternative)` row
of `df_long`, compute:

* `plevel1` = P(nest k(j) | i)
              = D_{k}^{τ_{k}} / Σ_m D_m^{τ_m}              (top level)
* `plevel2` = P(j | i) = P(j | k(j), i) · P(k(j) | i)      (joint, bottom)
* `p_cond`  = within-nest conditional P(j | k(j), i)
              = exp(V_ij / τ_{k}) / D_{k}                 (extra)

The mean of `plevel2` over rows where `<alt_var> == "beach"` is the
average predicted *unconditional* probability of choosing beach
across cases — and matches the empirical share within rounding.

`res` is the NamedTuple from `stata_nlogit`; the helper reads only
`β_alt`, `α`, `γ`, `τ`, `nests`, `nest_names`, `alt_specific`,
`case_vars`, `K` — fields every `stata_nlogit` fit exposes — and
reconstructs the alt → nest map from `res.nests`. `case_var` and
`alt_var` must match the long-form column names used at fit time.
"""
function _c15_nlogit_probs(res, df_long::DataFrames.AbstractDataFrame;
                      case_var::Symbol, alt_var::Symbol)
    # Reconstruct alt → nest index (1..K) from `res.nests`.
    nest_of_alt = Dict{String,Int}()
    for (k, pr) in enumerate(res.nests)
        aa = pr isa Pair ? pr.second : pr[2]
        for a in aa; nest_of_alt[String(a)] = k; end
    end
    K = res.K
    p_alt = length(res.alt_specific)
    p_csv = length(res.case_vars)

    altv = string.(df_long[!, alt_var])
    cs   = df_long[!, case_var]
    Z    = [Float64.(_c15_raw.(df_long[!, v])) for v in res.alt_specific]
    X    = [Float64.(_c15_raw.(df_long[!, v])) for v in res.case_vars]
    nest_idx = [nest_of_alt[a] for a in altv]
    N = DataFrames.nrow(df_long)

    function V_at(i)
        u = 0.0
        for c in 1:p_alt; u += res.β_alt[c] * Z[c][i]; end
        a = altv[i]
        u += res.α[a]
        for v in 1:p_csv; u += res.γ[a][v] * X[v][i]; end
        return u
    end

    # Group rows by case so we can compute D_k and the top-level sum.
    grp_rows = Dict{eltype(cs), Vector{Int}}()
    for i in 1:N; push!(get!(() -> Int[], grp_rows, cs[i]), i); end

    plevel1 = zeros(N)   # P(nest)   — Stata's top-level
    plevel2 = zeros(N)   # P(alt)    — Stata's joint at the bottom
    p_cond  = zeros(N)   # P(alt|nest) — within-nest conditional (extra)

    for (_, ridx) in grp_rows
        # log D_k via stable log-sum-exp
        sums  = zeros(K); maxes = fill(-Inf, K)
        Vi = Dict{Int,Float64}()
        for i in ridx
            k = nest_idx[i]
            v = V_at(i); Vi[i] = v
            vot = v / res.τ[res.nest_names[k]]
            if vot > maxes[k]; sums[k] *= exp(maxes[k] - vot); maxes[k] = vot; end
            sums[k] += exp(vot - maxes[k])
        end
        logD = [sums[k] > 0 ? maxes[k] + log(sums[k]) : -Inf for k in 1:K]
        # P(nest k) numerator: D_k^{τ_k} ∝ exp(τ_k logD_k)
        top  = [res.τ[res.nest_names[k]] * logD[k] for k in 1:K]
        tm   = maximum(top); log_top = tm + log(sum(exp.(top .- tm)))
        for i in ridx
            k = nest_idx[i]
            τ_k = res.τ[res.nest_names[k]]
            # P(j | k) = exp((V/τ_k) - logD_k)
            p_cond[i]  = exp(Vi[i] / τ_k - logD[k])
            # P(k)    = exp(τ_k logD_k - log_top)
            plevel1[i] = exp(top[k] - log_top)
            # P(j) = P(j|k) · P(k)
            plevel2[i] = p_cond[i] * plevel1[i]
        end
    end

    return (; plevel1, plevel2, p_cond)
end
