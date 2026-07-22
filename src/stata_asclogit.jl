import Optim

"""
    stata_asclogit(df_long; case_var, alt_var, depvar,
                   alt_specific=Symbol[], case_vars=Symbol[],
                   basealt=nothing, level=0.95, quiet=false)
        -> NamedTuple

Stata `asclogit <depvar> <alt_specific>, case(<case_var>) alternatives(<alt_var>) casevars(<case_vars>) basealternative(<basealt>) nolog`.

Alternative-specific conditional logit on long-form data — one row per
(case, alternative) — with utilities

    U_ij = αⱼ + Σ_c β_c · z_ijc + Σ_v γ_jv · x_iv + ε_ij

where:
* `z_ijc` (alt-specific) varies by both case and alternative; β_c is
  shared across alternatives.
* `x_iv` (case-specific, e.g. income) varies only by case; γ_jv is
  alternative-specific. Base alternative pins αⱼ ≡ 0 and γ_jv ≡ 0.

Choice probabilities are softmax over the J utilities per case.
Estimated by ML (LBFGS with autodiff gradient; OIM vcov via FD
Hessian).

Returns a NamedTuple with `β_alt`, `α` (Dict alt ⇒ α_j), `γ` (Dict
alt ⇒ Vector{γ_jv} matching `case_vars`), `β_cl` (the concatenated
parameter vector in `[alt_specific…; α_nonbase…; γ_v1_nonbase…;
γ_v2_nonbase…; …]` layout, used by `_c15_clogit_probs` and friends), `V`,
`se`, `ll`, `Wald`, `Wald_p`, `n_obs`, `n_cases`, `J`, `nonbase`,
`alts`.
"""
function stata_asclogit(df_long::DataFrames.AbstractDataFrame;
                        case_var::Symbol,
                        alt_var::Symbol,
                        depvar::Symbol,
                        alt_specific::AbstractVector{Symbol} = Symbol[],
                        case_vars::AbstractVector{Symbol} = Symbol[],
                        basealt = nothing,
                        level::Float64 = 0.95,
                        quiet::Bool = false)
    # Alternatives in their natural order (Stata uses the order they
    # first appear in the data; we replicate that).
    alts = String[]
    for v in df_long[!, alt_var]
        s = string(v)
        s ∈ alts || push!(alts, s)
    end
    J = length(alts)
    basealt === nothing && (basealt = alts[1])
    base_s = string(basealt)
    base_s ∈ alts || error("basealternative $basealt not in $alt_var")
    nonbase = filter(!=(base_s), alts)
    Jm1 = J - 1

    p_alt = length(alt_specific)        # alt-specific (shared β)
    p_csv = length(case_vars)            # case-specific (γ per alt)
    nparam = p_alt + Jm1 + Jm1 * p_csv

    # Per-row design slices (kept as Vectors keyed by alt string so the
    # MLE loop never has to look anything up by name).
    function as_float(col)
        return [Float64(_c15_raw(v)) for v in col]
    end
    y    = as_float(df_long[!, depvar])
    cs   = df_long[!, case_var]
    altv = [string(v) for v in df_long[!, alt_var]]
    Z    = [as_float(df_long[!, v]) for v in alt_specific]   # length p_alt
    X    = [as_float(df_long[!, v]) for v in case_vars]      # length p_csv
    N    = length(y)

    # Group rows by case_id (Stata calls these "cases" — one per
    # individual). We just take the unique cases in the order they
    # appear, then build a vector of row-index vectors.
    case_ids = unique(cs)
    G = length(case_ids)
    grp_rows = Dict{eltype(case_ids), Vector{Int}}()
    for i in 1:N
        push!(get!(() -> Int[], grp_rows, cs[i]), i)
    end

    # Index from non-base alt string → 1..(J-1) (parameter slot).
    nb_idx = Dict(s => k for (k, s) in enumerate(nonbase))

    function utility(theta, rowidx)
        # Build u_ij for a single (i, j) row given theta.
        u = zero(eltype(theta))
        for c in 1:p_alt
            u += theta[c] * Z[c][rowidx]
        end
        a = altv[rowidx]
        if a != base_s
            k = nb_idx[a]
            u += theta[p_alt + k]                       # α_j
            for v in 1:p_csv
                u += theta[p_alt + Jm1 + (v - 1) * Jm1 + k] * X[v][rowidx]
            end
        end
        return u
    end

    function negll(theta)
        ll = zero(eltype(theta))
        for (_, ridx) in grp_rows
            # Utilities for this case across all alternatives.
            us = [utility(theta, i) for i in ridx]
            mx = maximum(us)
            sm = log(sum(exp.(us .- mx))) + mx
            for i in ridx
                if y[i] > 0
                    ll += utility(theta, i) - sm
                end
            end
        end
        return -ll
    end

    θ0 = zeros(nparam)
    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 2000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    ll = -Optim.minimum(res)

    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0)
        f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x)
            xpi[i] += h_[i]; xmi[i] -= h_[i]
            H[i, i] = (f(xpi) - 2 * f0 + f(xmi)) / h_[i]^2
            for j in (i + 1):nθ
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
    V = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂)))
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    # Wald chi² on all slopes (matches Stata's header line).
    Wald   = θ̂' * (V \ θ̂)
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(nparam), Wald)

    # Unpack into Dict-style accessors for downstream callers.
    β_alt = [θ̂[c] for c in 1:p_alt]
    α     = Dict{String, Float64}(base_s => 0.0)
    γ     = Dict{String, Vector{Float64}}(base_s => zeros(p_csv))
    for (k, s) in enumerate(nonbase)
        α[s] = θ̂[p_alt + k]
        γ[s] = [θ̂[p_alt + Jm1 + (v - 1) * Jm1 + k] for v in 1:p_csv]
    end

    if !quiet
        function g9(x; w::Int = 10, sig::Int = 7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su = sig
            s = Printf.@sprintf("%.*g", su, x)
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

        # Alt-per-case stats (constant 4 for balanced; report nonetheless).
        alts_per_case = [length(grp_rows[c]) for c in case_ids]
        a_min = minimum(alts_per_case); a_max = maximum(alts_per_case)
        a_avg = sum(alts_per_case) / G

        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Alternative-specific conditional logit",
                       "Number of obs", commafmt(N))
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Case ID variable: $(case_var)",
                       "Number of cases", commafmt(G))
        println()
        Printf.@printf("%-56s%-18s = %d\n",
                       "Alternative variable: $(alt_var)",
                       "Alts per case: min", a_min)
        Printf.@printf("%-56s%-18s = %.1f\n", "", "avg", a_avg)
        Printf.@printf("%-56s%-18s = %d\n", "", "max", a_max)
        println()
        Printf.@printf("%56s%-13s = %6.2f\n", "", "Wald chi2($nparam)", Wald)
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        p_str  = Printf.@sprintf("%-13s = %6.4f", "Prob > chi2", Wald_p)
        pad    = max(0, 78 - length(ll_str) - length(p_str))
        println(ll_str, " "^pad, p_str)
        println()

        println("-"^78)
        crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)

        function _print_param(label, idx)
            b = θ̂[idx]; s = se[idx]
            z = b / s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit * s; hi = b + crit * s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w = 10), g9(s; w = 9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w = 9), g9(hi; w = 10))
        end

        # 1) Alt-specific block (header = alt_var, rows = alt-specific
        #    regressor names, shared β across alternatives).
        if p_alt > 0
            Printf.@printf("%-12s |\n", string(alt_var))
            for c in 1:p_alt
                _print_param(string(alt_specific[c]), c)
            end
        end

        # 2) One block per non-base alternative: case_vars + _cons.
        for (k, s_alt) in enumerate(nonbase)
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", s_alt)
            for v in 1:p_csv
                _print_param(string(case_vars[v]),
                             p_alt + Jm1 + (v - 1) * Jm1 + k)
            end
            _print_param("_cons", p_alt + k)
        end

        # 3) Base alternative row.
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |  (base alternative)\n", base_s)
        println("-"^78)
    end

    return (; θ̂, β_alt, α, γ, β_cl = θ̂, V, se,
              ll, Wald, Wald_p, nparam,
              n_obs = N, n_cases = G, J, alts, nonbase,
              alt_specific = collect(alt_specific),
              case_vars    = collect(case_vars),
              basealt      = base_s)
end

"""
    stata_clogit(df_long; case_var, depvar, regs, level=0.95, quiet=false)
        -> NamedTuple

Stata `clogit <depvar> <regs>, group(<case_var>)` — plain McFadden
conditional logit on long-form data with a flat regressor list (no
separate α_j; if you want alternative-specific intercepts pre-build
them as dummies and include them in `regs`). For each case i with
chosen alternative j*:

    log P(j*|i) = X_ij*·β − log Σ_l exp(X_il·β)

ML by `Optim.LBFGS` with autodiff gradient; OIM via FD Hessian. The
helper covers the simpler `clogit` case the way Stata's command does
it; for case-specific regressors with per-alternative γ_j you want
`stata_asclogit` instead.

Returns `(; β, V, se, ll, regs, n_obs, n_cases, nparam)`.
"""
function stata_clogit(df_long::DataFrames.AbstractDataFrame;
                          case_var::Symbol,
                          depvar::Symbol,
                          regs::AbstractVector{Symbol},
                          level::Float64 = 0.95,
                          quiet::Bool = false)
    cs = df_long[!, case_var]
    y  = [Float64(_c15_raw(v)) for v in df_long[!, depvar]]
    Z  = [[Float64(_c15_raw(v)) for v in df_long[!, r]] for r in regs]
    N  = length(y); nparam = length(regs)
    case_ids = unique(cs); G = length(case_ids)
    grp_rows = Dict{eltype(case_ids), Vector{Int}}()
    for i in 1:N; push!(get!(() -> Int[], grp_rows, cs[i]), i); end

    function negll(θ)
        ll = zero(eltype(θ))
        for (_, ridx) in grp_rows
            chosen = findfirst(==(1.0), [y[i] for i in ridx])
            chosen === nothing && continue
            η = [sum(θ[c] * Z[c][i] for c in 1:nparam) for i in ridx]
            mη = maximum(η)
            ll += η[chosen] - (mη + log(sum(exp(ηk - mη) for ηk in η)))
        end
        return -ll
    end

    res = _c15_optimize(negll, zeros(nparam), Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 4000);
                         autodiff = :forward)
    β  = Optim.minimizer(res); ll = -Optim.minimum(res)

    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:nθ
            xpi = copy(x); xmi = copy(x); xpi[i] += h_[i]; xmi[i] -= h_[i]
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
    V_cov = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, β)))
    se = sqrt.(max.(LinearAlgebra.diag(V_cov), 0.0))

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
                       "Conditional logit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Group variable: $(case_var)",
                       "Number of groups", commafmt(G))
        println(Printf.@sprintf("Log likelihood = %.4f", ll)); println()
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        for c in 1:nparam
            b = β[c]; s = se[c]; z = b/s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit*s; hi = b + crit*s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           string(regs[c]),
                           g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^78)
    end

    return (; β, V = V_cov, se, ll,
              regs = collect(regs), n_obs = N, n_cases = G, nparam)
end

"""
    stata_estat_alternatives(res, df_long; depvar=:d, alt_var=:fishmode,
                             case_var=:id)

Stata `estat alternatives` after `asclogit`. Per-alternative summary:
cases available, %cases, times chosen, and % chosen. Alternatives
appear in `res.alts` order (the order `stata_asclogit` first saw
them in the data).
"""
function stata_estat_alternatives(res, df_long;
                                  depvar::Symbol = :d,
                                  alt_var::Symbol = :fishmode,
                                  case_var::Symbol = :id)
    alts = collect(res.alts)
    cases_per_alt = Int[]
    times_chosen  = Int[]
    total_cases   = length(unique(df_long[!, case_var]))
    total_chosen  = 0
    for a in alts
        sub = view(df_long, df_long[!, alt_var] .== a, :)
        push!(cases_per_alt, length(unique(sub[!, case_var])))
        cnt = count(==(1.0), [Float64(_c15_raw(v)) for v in sub[!, depvar]])
        push!(times_chosen, cnt)
        total_chosen += cnt
    end
    _comma(n) = replace(string(n), r"(\d)(?=(\d{3})+$)" => s"\1,")

    println()
    println("Alternatives summary for $(alt_var)")
    println()
    println("  Index of  |  Cases   %Cases    #Times    %Times")
    println("alternative |  per alt per alt   chosen    chosen")
    println("-"^12, "+", "-"^43)
    for (i, a) in enumerate(alts)
        Printf.@printf("%11s | %7s  %6.1f   %7s   %6.2f\n",
                       a, _comma(cases_per_alt[i]),
                       100 * cases_per_alt[i] / total_cases,
                       _comma(times_chosen[i]),
                       100 * times_chosen[i] / total_chosen)
    end
    println("-"^12, "+", "-"^43)
    return nothing
end

"""
    stata_estat_mfx_asclogit(res, df_long; var, level=0.95)

Stata `estat mfx, varlist(<var>)` after `asclogit` for a single alt-
specific variable. At column means of every regressor (alt-specific:
one mean per alternative; case-specific: a single across-case mean),
reports the partial derivative

    ∂P̄_k / ∂x_j  =  P̄_k · (δ_kj − P̄_j) · β_var      (alt-specific var)

for every (target outcome k, varied alternative j) pair. Prints one
block per target outcome, matching Stata's "own" (k=j) / "cross"
(k≠j) effects. SE comes from a delta-method numerical Jacobian on
`res.β_cl` against `res.V`.
"""
function stata_estat_mfx_asclogit(res, df_long;
                                  var::Symbol,
                                  level::Float64 = 0.95,
                                  alt_var::Symbol = :fishmode)
    alts = collect(res.alts); J = res.J
    nonbase = collect(res.nonbase); base_s = res.basealt
    nb_idx  = Dict(s => k for (k, s) in enumerate(nonbase))
    p_alt   = length(res.alt_specific)
    p_csv   = length(res.case_vars)
    Jm1     = J - 1

    var_idx = findfirst(==(var), res.alt_specific)
    var_idx === nothing &&
        error("$var is not in res.alt_specific = $(res.alt_specific)")
    β_var_param_idx = var_idx

    z_alt = Dict{Tuple{Symbol,String}, Float64}()
    for v_sym in res.alt_specific, a in alts
        mask = string.(df_long[!, alt_var]) .== a
        z_alt[(v_sym, a)] =
            Statistics.mean(Float64(_c15_raw(v)) for v in df_long[mask, v_sym])
    end
    x_case = Dict{Symbol, Float64}(
        v => Statistics.mean(Float64(_c15_raw(x)) for x in df_long[!, v])
        for v in res.case_vars
    )

    function pbar(θ)
        η = zeros(eltype(θ), J)
        for (j, a) in enumerate(alts)
            u = zero(eltype(θ))
            for c in 1:p_alt
                u += θ[c] * z_alt[(res.alt_specific[c], a)]
            end
            if a != base_s
                k = nb_idx[a]
                u += θ[p_alt + k]
                for v in 1:p_csv
                    u += θ[p_alt + Jm1 + (v - 1) * Jm1 + k] *
                         x_case[res.case_vars[v]]
                end
            end
            η[j] = u
        end
        m = maximum(η); e = exp.(η .- m); return e ./ sum(e)
    end

    function me(θ, k_idx, j_idx)
        P = pbar(θ)
        δ = k_idx == j_idx ? 1.0 : 0.0
        return P[k_idx] * (δ - P[j_idx]) * θ[β_var_param_idx]
    end

    θ̂ = res.β_cl
    crit = Distributions.quantile(Distributions.Normal(),
                                  1 - (1 - level) / 2)
    h_step = sqrt(eps(Float64))

    function delta_se(f)
        g = zeros(length(θ̂))
        for i in eachindex(θ̂)
            s = max(abs(θ̂[i]) * h_step, h_step)
            tp = copy(θ̂); tp[i] += s
            tm = copy(θ̂); tm[i] -= s
            g[i] = (f(tp) - f(tm)) / (2 * s)
        end
        return sqrt(max(g' * res.V * g, 0.0))
    end

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

    println()
    println("Marginal effects after asclogit")
    Printf.@printf("Number of obs = %s\n", commafmt(res.n_obs))
    P_bar = pbar(θ̂)
    rows_all = NamedTuple[]
    for (k, ak) in enumerate(alts)
        println()
        Printf.@printf("y = Pr(%s==%s)  =  %.7f\n",
                       string(alt_var), ak, P_bar[k])
        println("-"^78)
        Printf.@printf("%14s |      dy/dx   Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "variable", 100 * level)
        println("-"^15, "+", "-"^62)
        for (j, aj) in enumerate(alts)
            kind = k == j ? "own" : "cross"
            pt = me(θ̂, k, j)
            se = delta_se(θ -> me(θ, k, j))
            z  = pt / se
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = pt - crit * se; hi = pt + crit * se
            push!(rows_all, (; target = ak, varied = aj, kind,
                              dydx = pt, se, z, p = pp, ci_lo = lo, ci_hi = hi))
            Printf.@printf("%6s %-7s | %s  %s  %s   %s    %s  %s\n",
                           string(var), aj,
                           g9(pt; w = 10), g9(se; w = 9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w = 9), g9(hi; w = 10))
        end
        println("-"^78)
    end
    return (; rows = rows_all, P_bar, z_alt, x_case)
end

"""
    _c15_clogit_probs(params, df_long, base="beach") -> Vector

Stata `predict, pr` after `asclogit`. Given the concatenated `β_cl`
parameter vector returned by `stata_asclogit` (layout: `[β_p; β_q;
α_nonbase…; γ_v1_nonbase…; γ_v2_nonbase…; …]` for the textbook's
`d ~ p q | income` setup), compute the per-(case, alternative)
predicted choice probability with a stable softmax over each case's
J utilities. Returns a vector aligned with `df_long`'s row order.
"""
function _c15_clogit_probs(params, df_long, base = "beach")
    alts = unique(df_long.fishmode)
    nonbase = filter(!=(base), alts)
    β_p = params[1]; β_q = params[2]
    α = Dict(zip(nonbase, params[3:5])); α[base] = 0.0
    γ = Dict(zip(nonbase, params[6:8])); γ[base] = 0.0
    probs = Float64[]
    for grp in DataFrames.groupby(df_long, :id)
        utils = β_p .* grp.p .+ β_q .* grp.q .+
                [α[a] for a in grp.fishmode] .+
                [γ[a] for a in grp.fishmode] .* grp.income
        max_u = maximum(utils)
        e = exp.(utils .- max_u)
        append!(probs, e ./ sum(e))
    end
    return probs
end
