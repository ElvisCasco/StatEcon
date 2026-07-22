# (deps provided by the StatEcon module)

"""
    stata_xtpoisson_pa(df; depvar, regs, idvar, timevar=nothing,
                       corr=:exchangeable, max_iter=20, tol=1e-7,
                       level=0.95, quiet=false) -> NamedTuple

Stata-style `xtpoisson <depvar> <regs>, pa corr(<corr>) vce(robust)` —
GEE (Liang & Zeger 1986) population-averaged Poisson estimator.
Working correlation `corr ∈ {:exchangeable, :independent, :unstructured}`.
For `:unstructured`, `timevar` MUST be provided so the helper can
identify each observation's time slot in the full T×T correlation
matrix; for unbalanced panels the relevant sub-matrix is sliced per
panel.

Pipeline (mirrors `stata_xtlogit_pa` but with the Poisson link):
  1. Pooled-Poisson warm start.
  2. Iterate until ‖Δβ‖ < tol:
     a. μ_i = exp(X_i β),  A_i = diag(μ_ij),  V_i = A_i^½ R(α) A_i^½
     b. Pearson residuals r_ij = (y_ij - μ_ij)/√μ_ij
     c. Estimate R from cross-time products of residuals
        - exchangeable: single ρ̂ = mean of r_is·r_it for s≠t
        - unstructured: full T×T pooled per (s, t) pair
     d. Newton update β ← β + (Σ Xᵢ' Aᵢ Vᵢ⁻¹ Aᵢ Xᵢ)⁻¹ Σ Xᵢ' Aᵢ Vᵢ⁻¹ (yᵢ − μᵢ)

Liang-Zeger sandwich SE: V = A⁻¹ B A⁻¹.

Output mirrors Stata's `xtpoisson, pa` block (Family: Poisson / Link:
Log / Correlation: <corr>) with the `(Std. err. adjusted for
clustering on <id>)` sub-note and the "Robust" sub-banner.
"""
function stata_xtpoisson_pa(df; depvar::Symbol,
                            regs::AbstractVector{Symbol},
                            idvar::Symbol,
                            timevar::Union{Nothing, Symbol} = nothing,
                            corr::Symbol = :exchangeable,
                            max_iter::Int = 20, tol::Float64 = 1e-7,
                            level::Float64 = 0.95, quiet::Bool = false)
    if corr == :unstructured && timevar === nothing
        error("corr = :unstructured requires `timevar` keyword")
    end

    needed = unique(vcat(depvar, idvar, regs))
    timevar !== nothing && (needed = unique(vcat(needed, timevar)))
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
    d = timevar === nothing ? DataFrames.sort(d, [idvar]) :
                              DataFrames.sort(d, [idvar, timevar])

    # Time-slot map (only used for :unstructured).
    unique_times = timevar === nothing ? Int[] :
                   sort(unique(d[!, timevar]))
    T_max_global = max(length(unique_times), 1)
    time_to_slot = timevar === nothing ?
                   Dict{Any, Int}() :
                   Dict(t => i for (i, t) in enumerate(unique_times))

    panels = DataFrames.groupby(d, idvar)
    pd = [(y = Float64.(g[!, depvar]),
           X = hcat([Float64.(g[!, v]) for v in regs]...,
                    ones(DataFrames.nrow(g))),
           t_slots = timevar === nothing ?
                     collect(1:DataFrames.nrow(g)) :
                     [time_to_slot[t] for t in g[!, timevar]])
          for g in panels]
    n_obs    = sum(length(p.y) for p in pd)
    n_panels = length(pd)
    k        = length(regs) + 1
    cnames   = vcat(string.(regs), "_cons")

    # Pooled-Poisson warm start.
    X_all = reduce(vcat, [p.X for p in pd])
    y_all = reduce(vcat, [p.y for p in pd])
    m_pooled = GLM.glm(X_all, y_all, Distributions.Poisson(), GLM.LogLink())
    β = GLM.coef(m_pooled)

    α      = 0.0
    R_full = Matrix(LinearAlgebra.I, T_max_global, T_max_global) .* 1.0

    function build_Ri(p)
        T_i = length(p.y)
        if corr == :independent
            return Matrix(LinearAlgebra.I, T_i, T_i) .* 1.0
        elseif corr == :exchangeable
            return [s == t ? 1.0 : α for s in 1:T_i, t in 1:T_i]
        elseif corr == :unstructured
            return R_full[p.t_slots, p.t_slots]
        else
            error("corr=$corr not supported")
        end
    end

    for iter in 1:max_iter
        # Pearson residuals.
        r_panels = Vector{NamedTuple}(undef, n_panels)
        for (i, p) in enumerate(pd)
            μ = exp.(p.X * β)
            σ = sqrt.(μ .+ 1e-12)
            r_panels[i] = (r = (p.y .- μ) ./ σ, t_slots = p.t_slots)
        end

        # Update working correlation.
        if corr == :exchangeable
            num, den = 0.0, 0
            for rp in r_panels
                T_i = length(rp.r)
                T_i < 2 && continue
                for s in 1:T_i, t in (s+1):T_i
                    num += rp.r[s] * rp.r[t]
                    den += 1
                end
            end
            α = den > 0 ? clamp(num / den, -0.99, 0.99) : 0.0
        elseif corr == :unstructured
            R_full = zeros(T_max_global, T_max_global)
            counts = zeros(Int, T_max_global, T_max_global)
            for rp in r_panels
                for i_in in eachindex(rp.r)
                    for j_in in eachindex(rp.r)
                        s = rp.t_slots[i_in]
                        t = rp.t_slots[j_in]
                        R_full[s, t] += rp.r[i_in] * rp.r[j_in]
                        counts[s, t] += 1
                    end
                end
            end
            for i in 1:T_max_global, j in 1:T_max_global
                if counts[i, j] > 0
                    R_full[i, j] /= counts[i, j]
                end
            end
            for i in 1:T_max_global
                R_full[i, i] = 1.0
            end
        end

        # Newton step on β.
        A_mat = zeros(k, k); rhs = zeros(k)
        for p in pd
            μ  = exp.(p.X * β)
            T_i = length(p.y)
            R_i = build_Ri(p)
            sqrtA = sqrt.(μ .+ 1e-12)
            Ainv  = LinearAlgebra.Diagonal(1 ./ sqrtA)
            Vi_inv = Ainv * LinearAlgebra.inv(R_i) * Ainv
            DX = LinearAlgebra.Diagonal(μ) * p.X
            A_mat .+= DX' * Vi_inv * DX
            rhs   .+= DX' * Vi_inv * (p.y .- μ)
        end
        β_new = β + A_mat \ rhs
        LinearAlgebra.norm(β_new - β) < tol && (β = β_new; break)
        β = β_new
    end

    # Sandwich SE (Liang-Zeger).
    A_mat = zeros(k, k); B = zeros(k, k)
    for p in pd
        μ  = exp.(p.X * β)
        T_i = length(p.y)
        R_i = build_Ri(p)
        sqrtA = sqrt.(μ .+ 1e-12)
        Ainv  = LinearAlgebra.Diagonal(1 ./ sqrtA)
        Vi_inv = Ainv * LinearAlgebra.inv(R_i) * Ainv
        DX = LinearAlgebra.Diagonal(μ) * p.X
        u  = p.y .- μ
        A_mat .+= DX' * Vi_inv * DX
        B     .+= DX' * Vi_inv * (u * u') * Vi_inv * DX
    end
    Ainv_full = LinearAlgebra.inv(A_mat)
    V  = Ainv_full * B * Ainv_full
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z  = β ./ se
    pv = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit  = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β .- crit .* se
    ci_hi = β .+ crit .* se

    slope_idx = 1:k - 1
    Wald   = β[slope_idx]' * LinearAlgebra.inv(V[slope_idx, slope_idx]) *
             β[slope_idx]
    Wald_p = 1 - Distributions.cdf(Distributions.Chisq(length(slope_idx)),
                                   Wald)

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
                   corr == :independent  ? "independent"  :
                   corr == :unstructured ? "unstructured" : string(corr)
        group_str = timevar === nothing ?
                    "Group variable: " * string(idvar) :
                    "Group and time vars: " * string(idvar) * " " *
                    string(timevar)
        println()
        Printf.@printf("%-53s%-17s= %6s\n",
                       "GEE population-averaged model",
                       "Number of obs", commafmt(n_obs))
        Printf.@printf("%-53s%-17s= %6s\n",
                       group_str,
                       "Number of groups", commafmt(n_panels))
        Printf.@printf("%-53s%s\n", "Family: Poisson", "Obs per group:  ")
        Printf.@printf("%-53s%18s = %6d\n", "Link:   Log", "min", T_min)
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

    return (; β, V, se, coefnames = cnames, α, R = R_full,
              n = n_obs, n_panels, T_min, T_max, T_avg,
              Wald, Wald_p, model = m_pooled)
end

