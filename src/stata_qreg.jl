## Stata-style quantile regression: `qreg` (single τ, iid Koenker–Bassett SEs)
## and `sqreg` (simultaneous QR across several τ with paired bootstrap SEs).
##
## Both use an IRLS / majorize–minimize scheme on the check-loss
##   ρ_τ(u) = u · (τ − 1{u < 0})
## which converges reliably and is fast enough for classroom-scale problems.

# ────────────────────────────────────────────────────────────────────────
# Internal IRLS core (used by both qreg and sqreg).
# Majorization: |u| ≤ u²/(2|u*|) + |u*|/2 ⇒ closed-form Newton step
#   β_new = (X'WX)⁻¹ [X'W y + (2τ−1) · X'1],  wᵢ = 1/max(|eᵢ|, δ)
# ────────────────────────────────────────────────────────────────────────
function _qr_irls(X::AbstractMatrix, Y::AbstractVector, τ::Real;
                  max_iter::Int = 5000, tol::Float64 = 1e-12)
    n     = length(Y)
    β     = X \ Y
    one_v = ones(n)
    δ     = 1e-2
    for _ in 1:max_iter
        e    = Y .- X * β
        δ    = max(min(δ, 0.5 * Statistics.median(abs.(e)) + 1e-12), 1e-10)
        w    = 1.0 ./ max.(abs.(e), δ)
        rhs  = X' * (w .* Y) .+ (2τ - 1) .* (X' * one_v)
        XtWX = X' * (w .* X)
        β_nw = XtWX \ rhs
        dif  = sqrt(sum((β_nw .- β).^2)) / max(1.0, sqrt(sum(β.^2)))
        β    = β_nw
        dif < tol && break
    end
    return β
end

_qr_rho(u, τ) = u >= 0 ? τ * u : (τ - 1) * u

# Stata-style thousand-comma formatter for `Number of obs` line
_qr_fmtn(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1,")

# Stata's %g rendering with leading-zero stripped for |x|<1
function _qr_g(x, w; sig::Int = 7)
    (ismissing(x) || !isfinite(x)) && return Printf.@sprintf("%*s", w, ".")
    s = Printf.@sprintf("%.*g", sig, x)
    0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1."))
    Printf.@sprintf("%*s", w, s)
end

"""
    stata_qreg(df, y, xs; τ=0.5, level=0.95, quiet=false)

Stata-style `qreg y xs…, quantile(τ)` — quantile regression via IRLS on the
check-loss. Standard errors use the Koenker–Bassett iid formula with the
Hall–Sheather bandwidth:

    ĥ  = n^{-1/3} · z_{α/2}^{2/3} · ((1.5 · φ(Φ⁻¹(τ))²) / (2·Φ⁻¹(τ)² + 1))^{1/3}
    ω̂  = (Q̂(τ + ĥ) − Q̂(τ − ĥ)) / (2ĥ)                (residual sparsity)
    s_τ = √(τ(1−τ)) · ω̂
    V   = s_τ² · (X'X)⁻¹

Returns a NamedTuple `(β, V, se, t, p, ci_lo, ci_hi, coefnames, n, MSD,
RSD, pseudo_R2, τ)`. Prints Stata's coefficient block unless `quiet=true`.
"""
function stata_qreg(df, y, xs::AbstractVector;
                    τ::Real = 0.5,
                    level::Float64 = 0.95,
                    max_iter::Int = 5000, tol::Float64 = 1e-12,
                    quiet::Bool = false)
    ys   = Symbol(y)
    xsv  = [Symbol(v) for v in xs]
    d    = DataFrames.dropmissing(df, unique(vcat(ys, xsv)))
    n    = DataFrames.nrow(d)
    Y    = Float64.(_sm_rawval.(d[!, ys]))
    X    = hcat([Float64.(_sm_rawval.(d[!, v])) for v in xsv]..., ones(n))
    k    = size(X, 2)

    β    = _qr_irls(X, Y, τ; max_iter, tol)
    û    = Y .- X * β

    MSD  = sum(_qr_rho(û[i], τ) for i in 1:n)
    q_y  = Statistics.quantile(Y, τ)
    RSD  = sum(_qr_rho(Y[i] - q_y, τ) for i in 1:n)
    R²ps = 1 - MSD / RSD

    # Hall–Sheather bandwidth + Koenker–Bassett sparsity → iid SE
    α    = 1 - level
    z_α  = Distributions.quantile(Distributions.Normal(), 1 - α/2)
    Φinv = Distributions.quantile(Distributions.Normal(), τ)
    φ_τ  = Distributions.pdf(Distributions.Normal(), Φinv)
    h    = n^(-1/3) * z_α^(2/3) *
           ((1.5 * φ_τ^2) / (2 * Φinv^2 + 1))^(1/3)
    p_up = min(τ + h, 1 - 1/n)
    p_dn = max(τ - h, 1/n)
    ω    = (Statistics.quantile(û, p_up) -
            Statistics.quantile(û, p_dn)) / (p_up - p_dn)
    s_τ  = sqrt(τ * (1 - τ)) * ω

    V    = s_τ^2 .* LinearAlgebra.inv(X' * X)
    se   = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    dofv  = n - k
    t_st  = β ./ se
    tcrit = Distributions.quantile(Distributions.TDist(dofv), 1 - α/2)
    pvals = 2 .* (1 .- Distributions.cdf.(Distributions.TDist(dofv), abs.(t_st)))
    ci_lo = β .- tcrit .* se
    ci_hi = β .+ tcrit .* se

    nms  = vcat(string.(xsv), "_cons")

    if !quiet
        title = τ == 0.5 ? "Median regression" :
                Printf.@sprintf("Quantile regression (τ = %.2f)", τ)
        println()
        header_tail = Printf.@sprintf("Number of obs = %10s", _qr_fmtn(n))
        println(title, " "^max(0, 52 - length(title)), header_tail)
        Printf.@printf("  Raw sum of deviations %.2f (about %g)\n", RSD, q_y)
        left3 = Printf.@sprintf("  Min sum of deviations %.3f", MSD)
        tail3 = Printf.@sprintf("Pseudo R2     = %10.4f", R²ps)
        println(left3, " "^max(0, 52 - length(left3)), tail3)
        println()

        lvl = round(Int, 100 * level)
        println("-"^78)
        Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                       string(ys), "Coefficient", "std. err.", "t", "P>|t|", lvl)
        println("-"^13, "+", "-"^64)
        for i in 1:k
            Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                           nms[i],
                           _qr_g(β[i], 10),
                           _qr_g(se[i], 9),
                           t_st[i], pvals[i],
                           _qr_g(ci_lo[i], 11),
                           _qr_g(ci_hi[i], 10))
        end
        println("-"^78)
    end

    return (; β, V, se, t = t_st, p = pvals, ci_lo, ci_hi,
              coefnames = nms, n, MSD, RSD, pseudo_R2 = R²ps, τ)
end

"""
    stata_sqreg(df, y, xs; τs=[0.25, 0.5, 0.75], reps=400,
                seed=nothing, level=0.95)

Stata-style `sqreg y xs…, q(τ₁ τ₂ …) reps(B)` — simultaneous quantile
regression at several τ values with paired-bootstrap standard errors.
Each replicate resamples rows once and re-fits every quantile on that
sample, preserving cross-quantile covariance (which is what makes `sqreg`
different from running `bsqreg` independently at each τ).

Returns `(; β, V, coefnames, β_by_q, se, t, p, ci_lo, ci_hi, pseudo_R2,
varnames, τs, n, reps, df_resid)`, where `β` and `V` are stacked across
quantiles with `coefnames = ["[qXX]var", …]` (compatible with cross-
quantile Wald tests via `stata_test_sureg`).
"""
function stata_sqreg(df, y, xs::AbstractVector;
                     τs::AbstractVector{<:Real} = [0.25, 0.50, 0.75],
                     reps::Int = 400,
                     seed::Union{Int,Nothing} = nothing,
                     level::Float64 = 0.95)
    ys  = Symbol(y)
    xsv = [Symbol(v) for v in xs]
    d   = DataFrames.dropmissing(df, unique(vcat(ys, xsv)))
    n   = DataFrames.nrow(d)
    k   = length(xsv) + 1
    Q   = length(τs)

    fits = [stata_qreg(d, ys, xsv; τ = Float64(τ), quiet = true) for τ in τs]
    β0   = [f.β for f in fits]

    seed !== nothing && Random.seed!(seed)
    β_boot = [Matrix{Float64}(undef, reps, k) for _ in 1:Q]
    for b in 1:reps
        idx = StatsBase.sample(1:n, n; replace = true)
        db  = d[idx, :]
        for (j, τ) in enumerate(τs)
            β_boot[j][b, :] =
                stata_qreg(db, ys, xsv; τ = Float64(τ), quiet = true).β
        end
    end
    se = [vec(Statistics.std(β_boot[j]; dims = 1)) for j in 1:Q]

    α     = 1 - level
    dofv  = n - k
    tcrit = Distributions.quantile(Distributions.TDist(dofv), 1 - α/2)
    t_q   = [β0[j] ./ se[j] for j in 1:Q]
    p_q   = [2 .* (1 .- Distributions.cdf.(Distributions.TDist(dofv), abs.(t_q[j])))
             for j in 1:Q]
    lo_q  = [β0[j] .- tcrit .* se[j] for j in 1:Q]
    hi_q  = [β0[j] .+ tcrit .* se[j] for j in 1:Q]

    nms = vcat(string.(xsv), "_cons")

    println()
    title = "Simultaneous quantile regression"
    Printf.@printf("%s%*sNumber of obs = %10s\n",
                   title, max(0, 52 - length(title)), "", _qr_fmtn(n))
    left_boot = Printf.@sprintf("  bootstrap(%d) SEs", reps)
    Printf.@printf("%s%*s.%02d Pseudo R2 = %10.4f\n",
                   left_boot, max(0, 52 - length(left_boot)), "",
                   round(Int, 100*τs[1]), fits[1].pseudo_R2)
    for j in 2:Q
        Printf.@printf("%52s.%02d Pseudo R2 = %10.4f\n",
                       "", round(Int, 100*τs[j]), fits[j].pseudo_R2)
    end
    println()

    lvl = round(Int, 100 * level)
    println("-"^78)
    Printf.@printf("%12s | %22s\n", "", "Bootstrap")
    Printf.@printf("%12s | %10s  %9s  %6s  %5s     [%d%% conf. interval]\n",
                   string(ys), "Coefficient", "std. err.", "t", "P>|t|", lvl)
    println("-"^13, "+", "-"^64)
    for j in 1:Q
        qlabel = Printf.@sprintf("q%02d", round(Int, 100*τs[j]))
        Printf.@printf("%-12s |\n", qlabel)
        for i in 1:k
            Printf.@printf("%12s | %s  %s  %7.2f  %6.3f  %s  %s\n",
                           nms[i],
                           _qr_g(β0[j][i], 10),
                           _qr_g(se[j][i], 9),
                           t_q[j][i], p_q[j][i],
                           _qr_g(lo_q[j][i], 11),
                           _qr_g(hi_q[j][i], 10))
        end
        j < Q && println("-"^13, "+", "-"^64)
    end
    println("-"^78)

    β_joint    = reduce(vcat, β0)
    boot_stack = reduce(hcat, β_boot)              # B × (Q·k)
    V_joint    = Statistics.cov(boot_stack)
    coefnames_joint = String[]
    for τ in τs, v in nms
        push!(coefnames_joint,
              "[" * Printf.@sprintf("q%02d", round(Int, 100*τ)) * "]" * v)
    end

    return (; β = β_joint, V = V_joint, coefnames = coefnames_joint,
              β_by_q = β0, se, t = t_q, p = p_q,
              ci_lo = lo_q, ci_hi = hi_q,
              pseudo_R2 = [f.pseudo_R2 for f in fits],
              varnames = nms, τs, n, reps, df_resid = dofv)
end
