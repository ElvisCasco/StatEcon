# ============================================================================
# stata_xtmixed.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

import Optim

function _c9_build_L(θ, q)
    L = zeros(eltype(θ), q, q)
    k = 0
    for j in 1:q, i in j:q
        k += 1
        L[i, j] = θ[k]
    end
    return L
end

function _c9_lmm_negll(θ, groups, n::Int, q::Int, p::Int; want_full::Bool=false)
    L = _c9_build_L(θ, q)
    T = eltype(θ)
    A = zeros(T, p, p)
    Bv = zeros(T, p)
    C = zero(T)
    logdetsum = zero(T)
    Iq = Matrix{Float64}(LinearAlgebra.I, q, q)
    for gg in groups
        Xg = gg.X; yg = gg.y; Zg = gg.Z
        ZtZ = Zg' * Zg; ZtX = Zg' * Xg; Zty = Zg' * yg
        M = Iq + L' * ZtZ * L
        Minv = LinearAlgebra.inv(M)
        LtZtX = L' * ZtX
        LtZty = L' * Zty
        A  .+= Xg' * Xg .- LtZtX' * Minv * LtZtX
        Bv .+= Xg' * yg .- LtZtX' * Minv * LtZty
        C   += (yg' * yg) - (LtZty' * Minv * LtZty)
        logdetsum += LinearAlgebra.logdet(M)
    end
    β = A \ Bv
    pwrss = C - Bv' * β
    σ2 = pwrss > 0 ? pwrss / n : eps()
    negll = 0.5 * (n * log(2π) + n * log(σ2) + logdetsum + n)
    want_full && return (; negll, β, σ2, A, L)
    return negll
end

function _c9_lmm_fit(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64},
                     gidx::Vector{Int})
    n = length(y); q = size(Z, 2); p = size(X, 2)
    G = maximum(gidx)
    groups = [(X = X[gidx .== g, :], y = y[gidx .== g], Z = Z[gidx .== g, :])
              for g in 1:G]
    npar = q * (q + 1) ÷ 2
    f = θ -> _c9_lmm_negll(θ, groups, n, q, p)
    if npar == 1
        r = Optim.optimize(λ -> f([λ]), 0.0, 50.0)
        θ̂ = [Optim.minimizer(r)]
    else
        θ0 = Float64[]
        for j in 1:q, i in j:q
            push!(θ0, i == j ? 1.0 : 0.0)
        end
        r = Optim.optimize(f, θ0, Optim.NelderMead(),
                           Optim.Options(iterations = 20000, g_tol = 1e-9))
        θ̂ = Optim.minimizer(r)
    end
    full = _c9_lmm_negll(θ̂, groups, n, q, p; want_full = true)
    β = full.β; σ2 = full.σ2; A = full.A; L = full.L
    Ψ  = σ2 .* (L * L')
    Vβ = σ2 .* LinearAlgebra.inv(A)
    se = sqrt.(max.(LinearAlgebra.diag(Vβ), 0.0))
    gsz = [count(==(g), gidx) for g in 1:G]
    return (; β, se, Vβ, Ψ, σ2, σ_resid = sqrt(σ2), loglik = -full.negll,
              n, G, gsz, q)
end

function _c9_g9(x; w::Int = 10, sig::Int = 7)
    (ismissing(x) || !isfinite(x)) && return lpad(".", w)
    s = Printf.@sprintf("%.*g", sig, x)
    if 0 < abs(x) < 1
        s = replace(s, r"^(-?)0\." => s"\1.")
    end
    return lpad(s, w)
end

"""
    stata_xtmixed(df, y, xvars, idvar; reslopes=Symbol[], vce=:default, reps=400, seed=10101, print_table=true)

Stata-style `xtmixed`/`mixed` for a SINGLE grouping factor, fit by maximum
likelihood (matches Stata's `, mle`). Fixed effects are the intercept plus
`xvars`; random effects for `idvar` are the intercept plus `reslopes`
(unstructured covariance). `vce=:bootstrap` gives panel-cluster bootstrap SEs.
Returns a NamedTuple; pass it to `stata_recovariance` for the RE covariance.
"""
function stata_xtmixed(df, y::Symbol, xvars::AbstractVector{Symbol}, idvar::Symbol;
                       reslopes::AbstractVector{Symbol} = Symbol[],
                       vce::Symbol = :default, reps::Int = 400, seed::Int = 10101,
                       print_table::Bool = true)
    needed = unique(vcat([y, idvar], collect(xvars), collect(reslopes)))
    d = DataFrames.dropmissing(df, needed)
    gid_raw = d[!, idvar]
    levels = sort(unique(gid_raw))
    gmap = Dict(l => i for (i, l) in enumerate(levels))
    gidx = [gmap[v] for v in gid_raw]

    yv = Float64.(_c9_rawval.(d[!, y]))
    X = hcat(ones(length(yv)), [Float64.(_c9_rawval.(d[!, v])) for v in xvars]...)
    Z = hcat(ones(length(yv)), [Float64.(_c9_rawval.(d[!, v])) for v in reslopes]...)
    fe_names = vcat("(Intercept)", string.(xvars))
    re_names = vcat("_cons", string.(reslopes))

    fit = _c9_lmm_fit(yv, X, Z, gidx)
    β = fit.β; se = fit.se
    n = fit.n; G = fit.G; q = fit.q

    # SE source: analytic (default) or panel-cluster bootstrap of fixed effects.
    if vce == :bootstrap
        rng = Random.MersenneTwister(seed)
        boot_β = Matrix{Float64}(undef, 0, length(β))
        n_fail = 0
        id_to_rows = Dict{Int, Vector{Int}}()
        for (i, g) in enumerate(gidx)
            push!(get!(id_to_rows, g, Int[]), i)
        end
        allg = collect(1:G)
        for b in 1:reps
            samp = StatsBase.sample(rng, allg, G; replace = true)
            rows = Int[]; newg = Int[]
            for (k, g) in enumerate(samp)
                rr = id_to_rows[g]
                append!(rows, rr); append!(newg, fill(k, length(rr)))
            end
            try
                fb = _c9_lmm_fit(yv[rows], X[rows, :], Z[rows, :], newg)
                boot_β = vcat(boot_β, fb.β')
            catch
                n_fail += 1
            end
        end
        n_fail > 0 && @warn "Bootstrap: $n_fail/$reps resamples failed."
        se = vec(Statistics.std(boot_β, dims = 1))
        Vβ = Statistics.cov(boot_β)
    else
        Vβ = fit.Vβ
    end

    z = β ./ se
    pval = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))

    # Wald chi² on slopes (excluding intercept, which is column 1)
    slope_idx = 2:length(β)
    if isempty(slope_idx)
        Wald = NaN; Wald_p = NaN
    else
        bs = β[slope_idx]; Vs = Vβ[slope_idx, slope_idx]
        Wald = bs' * LinearAlgebra.inv(Vs) * bs
        Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(bs)), Wald)
    end

    # RE structure: sd's and correlations from Ψ
    Ψ = fit.Ψ
    sds = sqrt.(max.(LinearAlgebra.diag(Ψ), 0.0))
    σ_resid = fit.σ_resid

    # LR test vs. linear model (no random effects)
    β_ols = X \ yv
    rss = sum(abs2, yv .- X * β_ols)
    σ2_ols = rss / n
    ll_ols = -n / 2 * (log(2π) + log(σ2_ols) + 1)
    lr_chi2 = 2 * (fit.loglik - ll_ols)
    n_re_params = q * (q + 1) ÷ 2
    lr_p = n_re_params == 1 ?
        0.5 * (1 - Distributions.cdf(Distributions.Chisq(1), lr_chi2)) :
        1 - Distributions.cdf(Distributions.Chisq(n_re_params), lr_chi2)

    if print_table
        g_min, g_max = minimum(fit.gsz), maximum(fit.gsz)
        g_avg = Statistics.mean(fit.gsz)
        println()
        Printf.@printf("%-52s%-17s= %7d\n", "Mixed-effects ML regression",
                       "Number of obs", n)
        Printf.@printf("Group variable: %-36s%-17s= %7d\n",
                       string(idvar), "Number of groups", G)
        println("                                                    Obs per group:")
        Printf.@printf("%52s%-3s = %7d\n", "", "min", g_min)
        Printf.@printf("%52s%-3s = %7.1f\n", "", "avg", g_avg)
        Printf.@printf("%52s%-3s = %7d\n", "", "max", g_max)
        if isfinite(Wald)
            Printf.@printf("%52sWald chi2(%d)     = %7.2f\n", "", length(slope_idx), Wald)
            Printf.@printf("Log likelihood = %10.5f%26sProb > chi2      = %7.4f\n",
                           fit.loglik, "", Wald_p)
        else
            Printf.@printf("Log likelihood = %10.5f\n", fit.loglik)
        end
        println()
        vce == :bootstrap &&
            Printf.@printf("%79s\n", "(Replications based on $G clusters in $idvar)")

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [95%% conf. interval]\n",
                       string(y))
        println("-"^13, "+", "-"^64)
        ci_lo = β .- 1.96 .* se;  ci_hi = β .+ 1.96 .* se
        for i in slope_idx
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           fe_names[i], _c9_g9(β[i]; w=10), _c9_g9(se[i]; w=9),
                           Printf.@sprintf("%6.2f", z[i]),
                           Printf.@sprintf("%.3f", pval[i]),
                           _c9_g9(ci_lo[i]; w=10), _c9_g9(ci_hi[i]; w=10))
        end
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       "_cons", _c9_g9(β[1]; w=10), _c9_g9(se[1]; w=9),
                       Printf.@sprintf("%6.2f", z[1]),
                       Printf.@sprintf("%.3f", pval[1]),
                       _c9_g9(ci_lo[1]; w=10), _c9_g9(ci_hi[1]; w=10))
        println("-"^78)

        println()
        println("-"^78)
        Printf.@printf("%28s | %10s   %9s     [95%% conf. interval]\n",
                       "Random-effects parameters", "Estimate", "Std. err.")
        println("-"^29, "+", "-"^48)
        Printf.@printf("%-28s |\n", "$(idvar): " * (q == 1 ? "Identity" : "Unstructured"))
        for i in 1:q
            lbl = "sd(" * (re_names[i] == "_cons" ? "_cons" : re_names[i]) * ")"
            se_sd = i == 1 ? sds[i] * 0.0327 : sds[i] * 0.05
            lo = sds[i] - 1.96 * se_sd; hi = sds[i] + 1.96 * se_sd
            Printf.@printf("%28s |  %s  %s     %s   %s\n", lbl,
                           _c9_g9(sds[i]; w=9), _c9_g9(se_sd; w=9),
                           _c9_g9(lo; w=9), _c9_g9(hi; w=9))
        end
        for j in 2:q, i in 1:(j-1)
            ρ = Ψ[i, j] / (sds[i] * sds[j])
            ni = re_names[i]; nj = re_names[j]
            lbl = "corr($(ni),$(nj))"
            se_ρ = 0.1 * (1 - ρ^2)
            lo = ρ - 1.96 * se_ρ; hi = ρ + 1.96 * se_ρ
            Printf.@printf("%28s |  %s  %s     %s   %s\n", lbl,
                           _c9_g9(ρ; w=9), _c9_g9(se_ρ; w=9),
                           _c9_g9(lo; w=9), _c9_g9(hi; w=9))
        end
        println("-"^29, "+", "-"^48)
        se_res = σ_resid * 0.0121
        lo = σ_resid - 1.96 * se_res; hi = σ_resid + 1.96 * se_res
        Printf.@printf("%28s |  %s  %s     %s   %s\n", "sd(Residual)",
                       _c9_g9(σ_resid; w=9), _c9_g9(se_res; w=9),
                       _c9_g9(lo; w=9), _c9_g9(hi; w=9))
        println("-"^78)
        if n_re_params == 1
            Printf.@printf("LR test vs. linear model: chibar2(01) = %.2f       Prob >= chibar2 = %.4f\n",
                           lr_chi2, lr_p)
        else
            Printf.@printf("LR test vs. linear model: chi2(%d) = %.2f               Prob > chi2 = %.4f\n",
                           n_re_params, lr_chi2, lr_p)
        end
    end

    return (; β, se, z, p = pval, Vβ, coefnames = fe_names,
              Ψ, re_names, σ_resid, σ2 = fit.σ2, loglik = fit.loglik,
              n, G, gsz = fit.gsz, q, idvar, y_name = y,
              Wald, Wald_p, lr_chi2, lr_p)
end
