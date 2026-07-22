# ============================================================================
# stata_gmm_poisson.jl — Stata nonlinear regression (Cameron & Trivedi ch10)
# ============================================================================

import Optim
import ForwardDiff

"""
    stata_gmm_poisson(df, depvar, regs, instruments; mean_fn=:exp,
                      onestep=true, level=0.95)

Stata's `gmm (y - mean_fn(xβ + b0)), instruments(z1 z2 …) [onestep]`.

Nonlinear-IV / GMM estimation of a model with conditional mean μ(x'β + b0);
the moment condition is `E[z · (y − μ(x'β + b0))] = 0`.

  - `depvar`      : dependent variable (Symbol).
  - `regs`        : regressors entering the linear index (`Vector{Symbol}`).
  - `instruments` : instrument variables (`Vector{Symbol}`).
  - `mean_fn`     : `:exp` (Poisson-mean) or `:identity`.
  - `onestep`     : `true` → identity W₁; `false` → two-step optimal W₂.

Just-identified (`#regs == #instruments`, `:exp`) is solved directly by
Newton–Raphson; the overidentified case minimizes the GMM objective with
`Optim` using an explicit `ForwardDiff` gradient. Returns a NamedTuple with
`β, se, V, Q, n, k, m, coefnames` and prints Stata-style output.
"""
function stata_gmm_poisson(df, depvar::Symbol, regs::Vector{Symbol},
                           instruments::Vector{Symbol};
                           mean_fn::Symbol=:exp, onestep::Bool=true,
                           level::Float64=0.95)
    needed = unique(vcat([depvar], regs, instruments))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing, Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    n = DataFrames.nrow(dfc)
    y  = Float64.(dfc[!, depvar])
    X  = hcat([Float64.(dfc[!, v]) for v in regs]...)         # n × k
    Z  = hcat([Float64.(dfc[!, v]) for v in instruments]...)  # n × m
    # Add constant to X (b0) and Z (_cons)
    X = hcat(X, ones(n))
    Z = hcat(Z, ones(n))
    k = size(X, 2)        # number of parameters (including b0)
    m_cnt = size(Z, 2)    # number of moment conditions

    # Mean function and its gradient w.r.t. β (anonymous closures capture the
    # local X — a named short-form method would hoist to module scope).
    μ_fn, ∂μ_fn = if mean_fn == :exp
        (β -> exp.(X * β)), (β -> exp.(X * β) .* X)
    elseif mean_fn == :identity
        (β -> X * β), (β -> X)
    else
        error("mean_fn=$mean_fn not supported")
    end

    # GMM objective: g(β)' W g(β) where g(β) = (1/n) Σ z_i (y_i - μ(x_i'β))
    function gmm_obj(β, W)
        u = y .- μ_fn(β)
        g = Z' * u ./ n
        return n * (g' * W * g)
    end

    # Step 1: identity weight matrix
    W1 = LinearAlgebra.I(m_cnt) |> Matrix

    if k == m_cnt && mean_fn == :exp
        # Just-identified: solve moment conditions via Newton-Raphson.
        β1 = zeros(k)
        try
            df_init = DataFrames.DataFrame(_y = y)
            for (j, v) in enumerate(regs)
                df_init[!, v] = X[:, j]
            end
            fml_init = StatsModels.term(:_y) ~ sum(StatsModels.term.(regs))
            m_init = GLM.glm(fml_init, df_init,
                             Distributions.Poisson(), GLM.LogLink())
            β_init_glm = GLM.coef(m_init)
            β1 = vcat(β_init_glm[2:end], β_init_glm[1])
        catch
        end
        for iter in 1:200
            μ = exp.(X * β1)
            r = Z' * (y .- μ)            # m-vector of moment values
            J = -(Z' * LinearAlgebra.Diagonal(μ) * X)
            Δ = try
                -(J \ r)
            catch
                -(LinearAlgebra.pinv(J) * r)
            end
            β1 = β1 + Δ
            if maximum(abs.(r)) < 1e-9 || maximum(abs.(Δ)) < 1e-12
                break
            end
        end
        u1 = y .- μ_fn(β1)
        g1 = Z' * u1 ./ n
        Q1 = n * (g1' * W1 * g1)
    else
        # Overidentified: minimize via Optim, warm-started from Poisson MLE.
        β_init = if mean_fn == :exp
            try
                df_init = DataFrames.DataFrame(_y = y)
                for (j, v) in enumerate(regs)
                    df_init[!, v] = X[:, j]
                end
                fml_init = StatsModels.term(:_y) ~ sum(StatsModels.term.(regs))
                m_init = GLM.glm(fml_init, df_init,
                                 Distributions.Poisson(), GLM.LogLink())
                β_init_glm = GLM.coef(m_init)
                vcat(β_init_glm[2:end], β_init_glm[1])
            catch
                zeros(k)
            end
        else
            zeros(k)
        end
        f1  = β -> gmm_obj(β, W1)
        g1! = (g, x) -> ForwardDiff.gradient!(g, f1, x)
        res1 = Optim.optimize(f1, g1!, β_init, Optim.LBFGS())
        β1 = Optim.minimizer(res1)
        Q1 = Optim.minimum(res1)
    end

    if onestep
        β_final = β1
        Q_final = Q1
    else
        # Step 2: optimal W = (1/n Σ z z' u²)⁻¹
        u1 = y .- μ_fn(β1)
        Ω1 = (Z' * LinearAlgebra.Diagonal(u1.^2) * Z) ./ n
        W2 = LinearAlgebra.inv(Ω1)
        f2  = β -> gmm_obj(β, W2)
        g2! = (g, x) -> ForwardDiff.gradient!(g, f2, x)
        res2 = Optim.optimize(f2, g2!, β1, Optim.BFGS())
        β_final = Optim.minimizer(res2)
        Q_final = Optim.minimum(res2)
    end

    # Robust sandwich VCE
    u_final = y .- μ_fn(β_final)
    μ_final = μ_fn(β_final)
    Dhat = -(Z' * LinearAlgebra.Diagonal(μ_final) * X)
    Shat = Z' * LinearAlgebra.Diagonal(u_final.^2) * Z
    if k == m_cnt
        V = (Dhat \ Shat) / Dhat'
    elseif onestep
        bread = LinearAlgebra.inv(Dhat' * Dhat)
        V = bread * (Dhat' * Shat * Dhat) * bread
    else
        V = LinearAlgebra.inv(Dhat' * LinearAlgebra.inv(Shat) * Dhat)
    end
    V = (V .+ V') ./ 2  # symmetrize numerical noise
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    z = β_final ./ se
    p = 2 .* (1 .- Distributions.cdf.(Distributions.Normal(), abs.(z)))
    crit = Distributions.quantile(Distributions.Normal(), 1 - (1-level)/2)
    ci_lo = β_final .- crit .* se
    ci_hi = β_final .+ crit .* se

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
        while i >= 1; push!(parts, s[max(1, i-2):i]); i -= 3; end
        return (num < 0 ? "-" : "") * join(reverse(parts), ",")
    end

    println()
    Printf.@printf("Final GMM criterion Q(b) = %.2e\n", Q_final / n)
    println()
    if k == m_cnt
        println("note: model is exactly identified.")
        println()
    end
    println("GMM estimation ")
    println()
    Printf.@printf("Number of parameters = %3d\n", k)
    Printf.@printf("Number of moments    = %3d\n", m_cnt)
    Printf.@printf("Initial weight matrix: Unadjusted%17sNumber of obs   = %10s\n",
                   "", commafmt(n))
    println()
    println("-"^78)
    println("             |               Robust")
    println("             | Coefficient  std. err.      z    P>|z|     [95% conf. interval]")
    println("-"^13, "+", "-"^64)
    for i in 1:(k-1)
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       string(regs[i]),
                       g9(β_final[i]; w=10), g9(se[i]; w=9),
                       Printf.@sprintf("%6.2f", z[i]),
                       Printf.@sprintf("%.3f", p[i]),
                       g9(ci_lo[i]; w=10), g9(ci_hi[i]; w=10))
    end
    println("-"^13, "+", "-"^64)
    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                   "/b0",
                   g9(β_final[end]; w=10), g9(se[end]; w=9),
                   Printf.@sprintf("%6.2f", z[end]),
                   Printf.@sprintf("%.3f", p[end]),
                   g9(ci_lo[end]; w=10), g9(ci_hi[end]; w=10))
    println("-"^78)
    println("Instruments for equation 1: ", join(instruments, " "), " _cons")

    return (; β=β_final, se, V, Q=Q_final, n, k, m=m_cnt,
              coefnames=vcat(string.(regs), "/b0"))
end
