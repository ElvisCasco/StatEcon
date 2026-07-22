# ============================================================================
# stata_fmm_poisson.jl — Stata `fmm: poisson` finite mixture (C&T ch17)
#
# The mixture log-likelihood is maximised by Optim.NelderMead (derivative-free:
# the log-sum-exp mixture surface is non-smooth for gradient methods, and the
# Poisson log-pmf's log-gamma term uses the ch17-shared `_c17_loggamma` defined
# in stata_ztnb.jl). vcov from a central-difference Hessian / score sandwich.
# ============================================================================

import Optim

"""
    stata_fmm_poisson(df, formula; ncomponents=2, vce=:robust, level=0.95,
                      quiet=false, seed_split=0.2) -> NamedTuple

Stata's `fmm <C>, [vce(robust)]: poisson <depvar> <regs>` — a C-component
Poisson finite mixture with constant (covariate-free) class
probabilities. The density is

    f(y|x) = Σ_c π_c · Poisson(y; μ_c),   μ_c = exp(x'β_c),

with class shares π_c parameterised by a reference-class multinomial
logit (class 1 = base): π_c ∝ exp(γ_c), γ_1 ≡ 0. Jointly MLE in
(β_1,…,β_C, γ_2,…,γ_C) by NelderMead (the log-sum-exp mixture surface
is non-smooth for gradient methods), warm-started by spreading the
Poisson MLE β across components (±`seed_split`). OIM vcov from a
central-difference Hessian; optional robust sandwich (`vce(robust)`).

Prints the Stata `fmm` layout — the latent-class `_cons` (logit) block
followed by one `Class : c / Response / Model: poisson` coefficient
block per component.

Returns `(; β_components, π, μ_components, γ, ll, V, coefnames, n, k,
β_pois_start, y, X)`.
"""
function stata_fmm_poisson(df, formula; ncomponents::Int = 2,
                          vce::Symbol = :robust, level::Float64 = 0.95,
                          quiet::Bool = false, seed_split::Float64 = 0.2)
    C = ncomponents
    needed = StatsModels.termvars(formula)
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    m_pois = GLM.glm(formula, dfc, Distributions.Poisson(), GLM.LogLink())
    βp = GLM.coef(m_pois)
    cn = GLM.coefnames(m_pois)
    y  = Float64.(GLM.response(m_pois))
    X  = GLM.modelmatrix(m_pois)
    n, k = size(X)
    nθ = C*k + (C-1)                       # β's + (C-1) logit class params

    # Unpack θ → (β_c matrix, class weights π).
    function _unpack(θ)
        B = reshape(view(θ, 1:C*k), k, C)              # column c = β_c
        γ = vcat(0.0, θ[(C*k+1):end])                  # γ_1 ≡ 0 (base)
        w = exp.(γ); π = w ./ sum(w)
        return B, π
    end

    function fmm_nll(θ)
        B, π = _unpack(θ)
        lπ = log.(π)
        ll = zero(eltype(θ))
        Eta = X * B                                     # n×C linear indices
        # Accumulate via log-sum-exp over components. Computing
        # `exp(logpmf)` then summing underflows to 0 for large counts
        # (docvis reaches 144), and a `max(·, eps())` floor would then
        # clamp the tail — silently dropping exactly the observations
        # that identify the high-mean component. logΣexp keeps the tail.
        for i in 1:n
            lc = lπ[1] + y[i]*Eta[i,1] - exp(Eta[i,1]) -
                 _c17_loggamma(y[i]+1)                  # log of weighted comp 1
            for c in 2:C
                lcc = lπ[c] + y[i]*Eta[i,c] - exp(Eta[i,c]) -
                      _c17_loggamma(y[i]+1)
                m   = max(lc, lcc)
                lc  = m + log(exp(lc - m) + exp(lcc - m))
            end
            ll += -lc
        end
        return ll
    end

    # Warm start: spread Poisson β across components, equal class shares.
    θ0 = Float64[]
    for c in 1:C
        push!(θ0, (βp .* (1 + seed_split*(2*(c-1)/(max(C-1,1)) - 1)))...)
    end
    append!(θ0, zeros(C-1))
    res = Optim.optimize(fmm_nll, θ0, Optim.NelderMead(),
                         Optim.Options(iterations = 8000, g_tol = 1e-8))
    θ̂  = Optim.minimizer(res)
    ll = -fmm_nll(θ̂)::Float64

    # Deterministic component order: sort by ascending fitted mean so the
    # labeling is reproducible and matches Stata's convention (class 1 =
    # lower mean). The likelihood is invariant to relabeling, so we just
    # rebuild θ̂ in sorted order and recompute the vcov below at the
    # sorted point. (Without this, two components with near-equal means
    # — common in NB mixtures — land in arbitrary order across runs.)
    let
        B0, π0 = _unpack(θ̂)
        means  = [Statistics.mean(exp.(X * B0[:, c])) for c in 1:C]
        ord    = sortperm(means)
        if ord != collect(1:C)
            Bs = B0[:, ord]; πs = π0[ord]
            γs = [log(πs[c] / πs[1]) for c in 2:C]
            θ̂  = vcat(vec(Bs), γs)
        end
    end

    # OIM / robust vcov via central-difference Hessian (+ score meat).
    function _fd_hessian(f, x)
        np = length(x); H = zeros(np, np)
        h  = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:np
            xpi = copy(x); xmi = copy(x); xpi[i] += h[i]; xmi[i] -= h[i]
            H[i, i] = (f(xpi) - 2*f0 + f(xmi)) / h[i]^2
            for j in (i+1):np
                xpp = copy(x); xpp[i]+=h[i]; xpp[j]+=h[j]
                xpm = copy(x); xpm[i]+=h[i]; xpm[j]-=h[j]
                xmp = copy(x); xmp[i]-=h[i]; xmp[j]+=h[j]
                xmm = copy(x); xmm[i]-=h[i]; xmm[j]-=h[j]
                H[i,j] = H[j,i] = (f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*h[i]*h[j])
            end
        end
        return H
    end
    V_oim = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hessian(fmm_nll, θ̂)))
    V = if vce == :robust
        function nll_i(θ, i)
            B, π = _unpack(θ); lπ = log.(π)
            ηi1 = LinearAlgebra.dot(view(X, i, :), view(B, :, 1))
            lc  = lπ[1] + y[i]*ηi1 - exp(ηi1) - _c17_loggamma(y[i]+1)
            for c in 2:C
                ηic = LinearAlgebra.dot(view(X, i, :), view(B, :, c))
                lcc = lπ[c] + y[i]*ηic - exp(ηic) - _c17_loggamma(y[i]+1)
                m   = max(lc, lcc); lc = m + log(exp(lc-m) + exp(lcc-m))
            end
            return -lc
        end
        hg = sqrt(sqrt(eps(Float64))) .* max.(abs.(θ̂), 1.0)
        S  = zeros(n, nθ)
        for i in 1:n, j in 1:nθ
            xp = copy(θ̂); xp[j]+=hg[j]; xm = copy(θ̂); xm[j]-=hg[j]
            S[i, j] = (nll_i(xp, i) - nll_i(xm, i)) / (2*hg[j])
        end
        V_oim * (S'*S) * V_oim * (n/(n-1))
    else
        V_oim
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    B̂, π̂ = _unpack(θ̂)
    β_components = [collect(B̂[:, c]) for c in 1:C]
    μ_components = [exp.(X * β_components[c]) for c in 1:C]
    γ̂ = θ̂[(C*k+1):end]

    function g9(x; w::Int=10, sig::Int=7)
        (ismissing(x) || !isfinite(x)) && return lpad(".", w)
        su=sig; s=Printf.@sprintf("%.*g", su, x)
        cap = (0 < abs(x) < 1 && x < 0) ? 10 : 9
        while length(s) > cap && su > 1; su-=1; s=Printf.@sprintf("%.*g", su, x); end
        0 < abs(x) < 1 && (s = replace(s, r"^(-?)0\." => s"\1.")); lpad(s, w)
    end
    commafmt(num) = begin
        s = string(abs(num)); parts = String[]; i = length(s)
        while i >= 1; push!(parts, s[max(1,i-2):i]); i-=3; end
        (num<0 ? "-" : "")*join(reverse(parts), ",")
    end
    crit = Distributions.quantile(Distributions.Normal(), 1-(1-level)/2)
    _rowfmt(lab, b, s) = begin
        z = b/s; pp = 2*(1-Distributions.cdf(Distributions.Normal(), abs(z)))
        Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n",
                       lab, g9(b;w=10), g9(s;w=9),
                       Printf.@sprintf("%7.2f", z), Printf.@sprintf("%.3f", pp),
                       g9(b-crit*s;w=9), g9(b+crit*s;w=10))
    end

    if !quiet
        depname = string(formula.lhs)
        ll_lab  = vce == :robust ? "Log pseudolikelihood" : "Log likelihood"
        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Finite mixture model", "Number of obs", commafmt(n))
        Printf.@printf("%s = %.4f\n\n", ll_lab, ll)
        se_lab = vce == :robust ? "std. err." : "Std. err."

        # Latent-class probability (multinomial logit) block.
        println("-"^78)
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       "", se_lab, 100*level)
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s | (base outcome)\n", "1.Class")
        for c in 2:C
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", "$(c).Class")
            idxγ = C*k + (c-1)
            _rowfmt("_cons", θ̂[idxγ], se[idxγ])
        end
        println("-"^78)

        # Per-class Poisson coefficient blocks.
        for c in 1:C
            println()
            Printf.@printf("Class       : %d\n", c)
            Printf.@printf("Response    : %s\n", depname)
            println("Model       : poisson")
            println()
            println("-"^78)
            Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                           depname, se_lab, 100*level)
            println("-"^13, "+", "-"^64)
            ci = findfirst(==("(Intercept)"), cn)
            sl = setdiff(1:k, ci === nothing ? Int[] : [ci])
            ord = vcat(sl, ci === nothing ? Int[] : [ci])
            for j in ord
                lab = cn[j] == "(Intercept)" ? "_cons" : cn[j]
                _rowfmt(lab, B̂[j, c], se[(c-1)*k + j])
            end
            println("-"^78)
        end
        Printf.@printf("\nClass shares: %s\n",
                       join((Printf.@sprintf("π%d = %.4f", c, π̂[c]) for c in 1:C),
                            "   "))
    end

    return (; β_components, π = π̂, μ_components, γ = γ̂, ll, V,
              coefnames = [x == "(Intercept)" ? "_cons" : x for x in cn],
              n, k, β_pois_start = βp, y, X)
end
