import Optim

"""
    stata_asmprobit(df_long; case_var, alt_var, depvar,
                    alt_specific=Symbol[], case_vars=Symbol[],
                    basealt=nothing, integration_pts=20,
                    level=0.95, quiet=false)
        -> NamedTuple

Stata `asmprobit <depvar> <alt_specific>, case(<c>) alternatives(<a>) casevars(<cv>)`,
**independent-errors** (`correlation(independent)`) variant. Same
regressor structure as `stata_asclogit` but with iid normal latent
errors. For chosen alternative j* in case i:

    P(j*|i) = ∫ ∏_{k ≠ j*} Φ(z + V_ij* − V_ik) φ(z) dz

with V_ij = Σ_c β_c · z_ijc + α_j + Σ_v γ_jv · x_iv (α_base = 0,
γ_jv = 0 at base). Integral by `integration_pts`-point Gauss–Hermite
(default 20). ML via LBFGS + autodiff gradient; OIM via FD Hessian.

**Caveats vs Stata's full `asmprobit, correlation(unstructured)`:**
this helper does NOT estimate an unstructured covariance matrix and
does NOT compute `vce(robust)` sandwich SEs. Treat results as a
sign/magnitude check, not a Stata-exact replication.
"""
function stata_asmprobit(df_long::DataFrames.AbstractDataFrame;
                         case_var::Symbol,
                         alt_var::Symbol,
                         depvar::Symbol,
                         alt_specific::AbstractVector{Symbol} = Symbol[],
                         case_vars::AbstractVector{Symbol}    = Symbol[],
                         basealt = nothing,
                         integration_pts::Int = 20,
                         level::Float64 = 0.95,
                         quiet::Bool = false)
    alts = String[]
    for v in df_long[!, alt_var]
        s = string(v); s ∈ alts || push!(alts, s)
    end
    J = length(alts)
    basealt === nothing && (basealt = alts[1])
    base_s = string(basealt)
    base_s ∈ alts || error("basealternative $basealt not in $alt_var")
    nonbase = filter(!=(base_s), alts)
    Jm1 = J - 1
    p_alt = length(alt_specific); p_csv = length(case_vars)
    nparam = p_alt + Jm1 + Jm1 * p_csv

    as_float(col) = [Float64(_c15_raw(v)) for v in col]
    y    = as_float(df_long[!, depvar])
    cs   = df_long[!, case_var]
    altv = [string(v) for v in df_long[!, alt_var]]
    Z    = [as_float(df_long[!, v]) for v in alt_specific]
    X    = [as_float(df_long[!, v]) for v in case_vars]
    N    = length(y)
    case_ids = unique(cs); G = length(case_ids)
    grp_rows = Dict{eltype(case_ids), Vector{Int}}()
    for i in 1:N; push!(get!(() -> Int[], grp_rows, cs[i]), i); end
    nb_idx = Dict(s => k for (k, s) in enumerate(nonbase))

    function utility(θ, rowidx)
        u = zero(eltype(θ))
        for c in 1:p_alt; u += θ[c] * Z[c][rowidx]; end
        a = altv[rowidx]
        if a != base_s
            k = nb_idx[a]
            u += θ[p_alt + k]
            for v in 1:p_csv
                u += θ[p_alt + Jm1 + (v - 1) * Jm1 + k] * X[v][rowidx]
            end
        end
        return u
    end

    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:(n_pts - 1)]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        return F.values, sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
    end
    gh_x, gh_w = _gauss_hermite(integration_pts)
    z_norm = sqrt(2.0) .* gh_x
    w_norm = gh_w ./ sqrt(Base.pi)
    Φ_(x) = 0.5 * (1 + _c15_erf(x / sqrt(2.0)))

    function negll(θ)
        ll = zero(eltype(θ))
        for (_, ridx) in grp_rows
            V = [utility(θ, i) for i in ridx]
            chosen_local = findfirst(==(1.0), [y[i] for i in ridx])
            chosen_local === nothing && continue
            v_chosen = V[chosen_local]
            P_i = zero(eltype(θ))
            for g in eachindex(z_norm)
                z = z_norm[g]; pg = one(eltype(θ))
                for j in eachindex(V)
                    j == chosen_local && continue
                    pg *= Φ_(z + v_chosen - V[j])
                end
                P_i += w_norm[g] * pg
            end
            ll += log(max(P_i, eps(Float64)))
        end
        return -ll
    end

    res = _c15_optimize(negll, zeros(nparam), Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 4000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res); ll = -Optim.minimum(res)

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
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h_[i]*h_[j])
            end
        end
        return H
    end
    V_cov = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂)))
    se = sqrt.(max.(LinearAlgebra.diag(V_cov), 0.0))

    β_alt = [θ̂[c] for c in 1:p_alt]
    α     = Dict{String,Float64}(base_s => 0.0)
    γ     = Dict{String,Vector{Float64}}(base_s => zeros(p_csv))
    for (k, s) in enumerate(nonbase)
        α[s] = θ̂[p_alt + k]
        γ[s] = [θ̂[p_alt + Jm1 + (v - 1) * Jm1 + k] for v in 1:p_csv]
    end

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
                       "Alternative-specific multinomial probit (iid)",
                       "Number of obs", commafmt(N))
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Case ID variable: $(case_var)",
                       "Number of cases", commafmt(G))
        Printf.@printf("%-56s%-13s = %6d\n",
                       "Alternative variable: $(alt_var)",
                       "Integration pts.", integration_pts)
        println(Printf.@sprintf("Log likelihood = %.4f", ll))
        println("Note: `correlation(independent)` approximation — not Stata's `unstructured`.")
        println()
        function _print(label, idx)
            b = θ̂[idx]; s = se[idx]; z = b/s
            pp = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
            lo = b - crit*s; hi = b + crit*s
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(b; w=10), g9(s; w=9),
                           Printf.@sprintf("%7.2f", z),
                           Printf.@sprintf("%.3f", pp),
                           g9(lo; w=9), g9(hi; w=10))
        end
        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        if p_alt > 0
            Printf.@printf("%-12s |\n", string(alt_var))
            for c in 1:p_alt
                _print(string(alt_specific[c]), c)
            end
        end
        for (k, s_alt) in enumerate(nonbase)
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", s_alt)
            for v in 1:p_csv
                _print(string(case_vars[v]),
                       p_alt + Jm1 + (v - 1) * Jm1 + k)
            end
            _print("_cons", p_alt + k)
        end
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s |  (base alternative)\n", base_s)
        println("-"^78)
    end

    return (; β_alt, α, γ, θ̂, V = V_cov, se, ll,
              n_obs = N, n_cases = G, J, alts, nonbase, basealt = base_s,
              alt_specific = collect(alt_specific),
              case_vars    = collect(case_vars),
              nparam, integration_pts)
end
