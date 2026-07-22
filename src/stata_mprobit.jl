import Optim

"""
    stata_mprobit(df, depvar, regs; baseoutcome=nothing, integration_pts=20,
                  level=0.95, quiet=false)
        -> NamedTuple

Stata `mprobit <depvar> <regs>, baseoutcome(<v>)`. Independent-errors
multinomial probit on case-specific regressors. For each obs i with
chosen alternative j*:

    P(j*|i) = ∫ ∏_{k ≠ j*} Φ(z + V_ij* − V_ik) φ(z) dz

with V_ik = X_i · β_k and β_{base} = 0. The integral is over the
latent error of the chosen alternative; we compute it by Gauss–
Hermite quadrature (`integration_pts` nodes, default 20 — Stata's
default is 15).

Parametrisation: design `X = [1, regs…]` (intercept first), parameter
matrix `B` is `K × (J−1)` with columns ordered to match the non-base
alternatives in sorted order. Estimation by ML (`Optim.LBFGS` with
autodiff gradient); OIM vcov via FD Hessian.

Returns `(; B, V, se, SE, X, J, dfc, depvar_name, regs, base_value,
base_pos, cat_labels, ll, ll_null, LR, LR_p, pseudo_r2, n, nparam,
integration_pts)`.
"""
function stata_mprobit(df::DataFrames.AbstractDataFrame, depvar::Symbol,
                      regs::AbstractVector{Symbol};
                      baseoutcome = nothing,
                      integration_pts::Int = 20,
                      level::Float64 = 0.95,
                      quiet::Bool = false)
    cols = vcat([depvar], collect(regs))
    dfc  = DataFrames.dropmissing(df[:, cols])
    for c in cols
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(_c15_raw.(col))
        end
    end
    y_raw = [_c15_raw(v) for v in dfc[!, depvar]]
    cats  = sort(unique(y_raw)); J = length(cats)
    base_val = baseoutcome === nothing ? cats[1] : baseoutcome
    base_pos = findfirst(==(base_val), cats)
    base_pos === nothing && error("baseoutcome $base_val not in $depvar")
    others_pos = [j for j in 1:J if j != base_pos]
    cat_labels = String[]
    for c in cats
        idx = findfirst(==(c), y_raw); v = dfc[idx, depvar]
        push!(cat_labels,
              hasproperty(v, :labels) && hasproperty(v, :value) &&
              haskey(v.labels, v.value) ? v.labels[v.value] : string(c))
    end
    y_idx = [findfirst(==(yi), cats) for yi in y_raw]
    N = length(y_idx)
    X = hcat(ones(N), [Float64.(_c15_raw.(dfc[!, r])) for r in regs]...)
    K = size(X, 2)
    coefnames = ["_cons", string.(regs)...]
    nparam = K * (J - 1)

    # Gauss–Hermite (∫ f(x) e^(−x²) dx) via Golub–Welsch.
    function _gauss_hermite(n_pts::Int)
        a = zeros(n_pts)
        b = [sqrt(j / 2) for j in 1:(n_pts - 1)]
        T = LinearAlgebra.SymTridiagonal(a, b)
        F = LinearAlgebra.eigen(T)
        nodes   = F.values
        weights = sqrt(Base.pi) .* (F.vectors[1, :]) .^ 2
        return nodes, weights
    end
    gh_x, gh_w = _gauss_hermite(integration_pts)
    # ∫ f(z) φ(z) dz ≈ Σ_g (w_g / √π) f(√2 · x_g)
    inv_sqrt_π = 1 / sqrt(Base.pi)
    z_norm = sqrt(2.0) .* gh_x        # GH nodes scaled for φ(z) measure
    w_norm = gh_w .* inv_sqrt_π

    # Standard normal CDF (avoids depending on Distributions inside negll).
    Φ(x) = 0.5 * (1 + _c15_erf(x / sqrt(2.0)))

    function negll(θ)
        B = reshape(θ, K, J - 1)
        η = X * B                                # N × (J-1), V_i,others
        ll = zero(eltype(θ))
        for i in 1:N
            yi = y_idx[i]
            # V_ij (length J): base = 0, others = η[i, k]
            V = zeros(eltype(θ), J)
            for k in 1:(J-1); V[others_pos[k]] = η[i, k]; end
            v_chosen = V[yi]
            P_i = zero(eltype(θ))
            for g in eachindex(z_norm)
                z = z_norm[g]
                pg = one(eltype(θ))
                for j in 1:J
                    j == yi && continue
                    pg *= Φ(z + v_chosen - V[j])
                end
                P_i += w_norm[g] * pg
            end
            ll += log(max(P_i, eps(Float64)))
        end
        return -ll
    end

    # Warm start: zeros (degenerate point is well-defined under MNP).
    θ0 = zeros(nparam)
    res = _c15_optimize(negll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8, iterations = 4000);
                         autodiff = :forward)
    θ̂  = Optim.minimizer(res)
    B  = reshape(θ̂, K, J - 1)
    ll = -Optim.minimum(res)

    function _fd_hessian(f, x)
        nθ = length(x); H = zeros(nθ, nθ)
        h_ = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
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
    V = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(negll, θ̂)))
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    SE = reshape(se, K, J - 1)

    counts = [count(==(j), y_idx) for j in 1:J]
    ll_null = sum(c * log(c / N) for c in counts if c > 0)
    LR = 2 * (ll - ll_null)
    df_lr = (K - 1) * (J - 1)
    LR_p = 1 - Distributions.cdf(Distributions.Chisq(df_lr), LR)
    pseudo_r2 = 1 - ll / ll_null

    if !quiet
        crit = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
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
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Multinomial probit regression",
                       "Number of obs", commafmt(N))
        Printf.@printf("%56s%-13s = %6d\n", "",
                       "Integration pts.", integration_pts)
        Printf.@printf("%56s%-13s = %6.2f\n", "",
                       "Wald chi2($(df_lr))", LR)
        Printf.@printf("%56s%-13s = %6.4f\n", "", "Prob > chi2", LR_p)
        ll_str = Printf.@sprintf("Log likelihood = %.4f", ll)
        r2_str = Printf.@sprintf("%-13s = %6.4f", "Pseudo R2", pseudo_r2)
        pad = max(0, 78 - length(ll_str) - length(r2_str))
        println(ll_str, " "^pad, r2_str); println()

        println("-"^78)
        Printf.@printf("%12s | Coefficient  Std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       string(depvar), 100 * level)
        println("-"^13, "+", "-"^64)
        for (i, j) in enumerate(1:J)
            Printf.@printf("%-12s |\n", cat_labels[j])
            if j == base_pos
                Printf.@printf("%12s |  (base outcome)\n", "")
            else
                k = findfirst(==(j), others_pos)
                for r in vcat(2:K, [1])
                    label = r == 1 ? "_cons" : coefnames[r]
                    b  = B[r, k]; s = SE[r, k]; z = b/s
                    p  = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
                    lo = b - crit*s; hi = b + crit*s
                    Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                                   label, g9(b; w=10), g9(s; w=9),
                                   Printf.@sprintf("%7.2f", z),
                                   Printf.@sprintf("%.3f", p),
                                   g9(lo; w=9), g9(hi; w=10))
                end
            end
            i < J && println("-"^13, "+", "-"^64)
        end
        println("-"^78)
        Printf.@printf("Base outcome = %s (%s)\n",
                       string(base_val), cat_labels[base_pos])
    end

    return (; B, V, se, SE, X, J, dfc,
              depvar_name = depvar, regs = collect(regs),
              base_value = base_val, base_pos, cat_labels,
              ll, ll_null, LR, LR_p, pseudo_r2,
              n = N, nparam, integration_pts)
end
