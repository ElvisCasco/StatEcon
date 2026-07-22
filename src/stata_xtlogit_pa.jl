# (deps provided by the StatEcon module)
import Optim
import ForwardDiff

# ── shared ch18 helpers (defined once; referenced across ch18 files) ──────
# _c18_loggamma: log Γ(x) via the Lanczos g=7 approximation (with reflection
# for x < 0.5). Dependency-free (no SpecialFunctions) and ForwardDiff-safe,
# so it can be used inside autodiff log-likelihoods.
function _c18_loggamma(x::Real)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(pi / sin(pi * x)) - _c18_loggamma(1 - x)   # reflection
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 1:8
        a += c[i + 1] / (x + i)
    end
    return 0.5 * log(2pi) + (x + 0.5) * log(t) - t + log(a)
end

# _c18_optimize: replacement for the notebook's _optimize_compat. The
# installed Optim's autodiff=:forward kwarg is broken, so gradient-based
# methods get an explicit ForwardDiff gradient via the 5-arg call form;
# derivative-free NelderMead is passed through unchanged.
function _c18_optimize(f, x0, method, opts)
    if method isa Optim.NelderMead
        return Optim.optimize(f, x0, method, opts)
    end
    g!(G, x) = ForwardDiff.gradient!(G, f, x)
    return Optim.optimize(f, g!, x0, method, opts)
end

"""
    stata_xtlogit_pa(df; depvar, regs, idvar, corr=:exchangeable,
                     max_iter=20, tol=1e-7, level=0.95, quiet=false)
        -> NamedTuple

Stata-style `xtlogit <depvar> <regs>, pa corr(<corr>) vce(robust) nolog` —
GEE (Liang & Zeger 1986) population-averaged logit estimator with
working correlation `corr ∈ {:exchangeable, :independent}`.

Algorithm (logit GEE):
  1. Pooled-logit warm start.
  2. Iterate until ‖Δβ‖ < tol:
     a. μ_i = Λ(X_i β),  A_i = diag(μ_ij(1-μ_ij)),  V_i = A_i^½ R(α) A_i^½
     b. Pearson residuals  r_ij = (y_ij - μ_ij)/√(μ_ij(1-μ_ij))
     c. For exchangeable: α̂ = mean of  r_is·r_it  for s≠t  across panels
        (independent: α̂ = 0)
     d. β ← β + (Σ X_i' A_i V_i⁻¹ A_i X_i)⁻¹ Σ X_i' A_i V_i⁻¹ (y_i - μ_i)

Liang-Zeger sandwich SE:
  V_β = A⁻¹ B A⁻¹,  A = Σ X_i' A_i V_i⁻¹ A_i X_i,
                    B = Σ X_i' A_i V_i⁻¹ u_i u_i' V_i⁻¹ A_i X_i.

Output mirrors Stata's `xtlogit, pa` block:
  - Header (Number of obs / Number of groups / Family / Link / Correlation /
    Obs per group min/avg/max / Wald chi² / Prob > chi² / Scale parameter).
  - "(Std. err. adjusted for clustering on <id>)" sub-note.
  - Coefficient table with "Robust" sub-banner.

Returns `(; β, V, se, coefnames, α, n, n_panels, T_min, T_avg, T_max,
Wald, Wald_p, model)`.
"""
function stata_xtlogit_pa(df; depvar::Symbol,
                          regs::AbstractVector{Symbol},
                          idvar::Symbol,
                          corr::Symbol = :exchangeable,
                          max_iter::Int = 20, tol::Float64 = 1e-7,
                          level::Float64 = 0.95, quiet::Bool = false)
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

    panels = DataFrames.groupby(d, idvar)
    pd = [(y = Float64.(g[!, depvar]),
           X = hcat([Float64.(g[!, v]) for v in regs]...,
                    ones(DataFrames.nrow(g))))     # slopes first, _cons last
          for g in panels]
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    # Initial pooled-logit warm start.
    X_all = reduce(vcat, [p.X for p in pd])
    y_all = reduce(vcat, [p.y for p in pd])
    m_pooled = GLM.glm(X_all, y_all, Distributions.Binomial(), GLM.LogitLink())
    β = GLM.coef(m_pooled)

    σmu(η) = (μ = 1 / (1 + exp(-η)); μ * (1 - μ))

    α = 0.0
    for iter in 1:max_iter
        # Pearson residuals
        r_panels = Vector{Vector{Float64}}(undef, n_panels)
        for (i, p) in enumerate(pd)
            η = p.X * β
            μ = 1 ./ (1 .+ exp.(-η))
            σ = sqrt.(μ .* (1 .- μ) .+ 1e-12)
            r_panels[i] = (p.y .- μ) ./ σ
        end
        # Estimate exchangeable α
        α = if corr == :exchangeable
            num, den = 0.0, 0
            for r in r_panels
                T_i = length(r)
                T_i < 2 && continue
                for s in 1:T_i, t in (s+1):T_i
                    num += r[s] * r[t]
                    den += 1
                end
            end
            den > 0 ? clamp(num / den, -0.99, 0.99) : 0.0
        else
            0.0
        end

        # Newton step on β
        A = zeros(k, k); b = zeros(k)
        for p in pd
            η  = p.X * β
            μ  = 1 ./ (1 .+ exp.(-η))
            μp = μ .* (1 .- μ) .+ 1e-12       # link derivative
            T_i = length(p.y)
            R = corr == :exchangeable ?
                [s == t ? 1.0 : α for s in 1:T_i, t in 1:T_i] :
                Matrix(LinearAlgebra.I, T_i, T_i)
            sqrtA = sqrt.(μp)
            Ainv  = LinearAlgebra.Diagonal(1 ./ sqrtA)
            Vi_inv = Ainv * LinearAlgebra.inv(R) * Ainv
            DX = LinearAlgebra.Diagonal(μp) * p.X
            A   .+= DX' * Vi_inv * DX
            b   .+= DX' * Vi_inv * (p.y .- μ)
        end
        β_new = β + A \ b
        LinearAlgebra.norm(β_new - β) < tol && (β = β_new; break)
        β = β_new
    end

    # Sandwich SE (Liang-Zeger)
    A = zeros(k, k); B = zeros(k, k)
    for p in pd
        η  = p.X * β
        μ  = 1 ./ (1 .+ exp.(-η))
        μp = μ .* (1 .- μ) .+ 1e-12
        T_i = length(p.y)
        R = corr == :exchangeable ?
            [s == t ? 1.0 : α for s in 1:T_i, t in 1:T_i] :
            Matrix(LinearAlgebra.I, T_i, T_i)
        sqrtA = sqrt.(μp)
        Ainv  = LinearAlgebra.Diagonal(1 ./ sqrtA)
        Vi_inv = Ainv * LinearAlgebra.inv(R) * Ainv
        DX = LinearAlgebra.Diagonal(μp) * p.X
        u  = p.y .- μ
        A .+= DX' * Vi_inv * DX
        B .+= DX' * Vi_inv * (u * u') * Vi_inv * DX
    end
    Ainv_full = LinearAlgebra.inv(A)
    V  = Ainv_full * B * Ainv_full
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    # Wald chi² on slopes (excludes _cons = last position)
    slope_idx = 1:k-1
    Wald   = β[slope_idx]' * LinearAlgebra.inv(V[slope_idx, slope_idx]) * β[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)), Wald)

    T_per = [length(p.y) for p in pd]
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
        corr_str = corr == :exchangeable ? "exchangeable" :
                   corr == :independent  ? "independent" : string(corr)
        println()
        Printf.@printf("%-53s%-17s= %6s\n",
                       "GEE population-averaged model",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Group variable: " * string(idvar),
                       "Number of groups", commafmt(n_panels))
        Printf.@printf("%-53s%s\n",
                       "Family: Binomial", "Obs per group:  ")
        Printf.@printf("%-53s%18s = %6d\n",
                       "Link:   Logit", "min", T_min)
        Printf.@printf("%-53s%18s = %6.1f\n",
                       "Correlation: " * corr_str, "avg", T_avg)
        Printf.@printf("%-53s%18s = %6d\n", "", "max", T_max)
        Printf.@printf("%-53s%-17s= %6s\n", "",
                       "Wald chi2($(length(slope_idx)))",
                       Printf.@sprintf("%.2f", Wald))
        Printf.@printf("%-53s%-17s= %6s\n",
                       "Scale parameter = 1",
                       "Prob > chi2", Printf.@sprintf("%.4f", Wald_p))
        println()
        cl_str = "(Std. err. adjusted for clustering on $idvar)"
        println(lpad(cl_str, 78))
        println("-"^78)
        println("             |               Robust")
        Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100*level)
        println("-"^13, "+", "-"^64)
        for i in vcat(collect(slope_idx), [k])
            label = i == k ? "_cons" : cnames[i]
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                           label, g9(β[i]; w=10), g9(se[i]; w=9),
                           Printf.@sprintf("%7.2f", z[i]),
                           Printf.@sprintf("%.3f", pv[i]),
                           g9(ci_lo[i]; w=9), g9(ci_hi[i]; w=10))
        end
        println("-"^78)
    end

    return (; β, V, se, coefnames = cnames, α, n = n_obs, n_panels,
              T_min, T_max, T_avg, Wald, Wald_p, model = m_pooled)
end

