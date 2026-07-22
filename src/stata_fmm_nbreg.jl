# ============================================================================
# stata_fmm_nbreg.jl — Stata `fmm: nbreg` finite mixture (C&T ch17)
#
# NB2-component mixture, fit by Optim.NelderMead (derivative-free; the NB2
# log-pmf's log-gamma terms use the ch17-shared `_c17_loggamma` from
# stata_ztnb.jl and log-sum-exp keeps the long count tail). Robust vcov from a
# central-difference Hessian + score sandwich.
# ============================================================================

import Optim

"""
    stata_fmm_nbreg(df, formula; ncomponents=2, level=0.95, quiet=false,
                    seed_split=0.1) -> NamedTuple

Stata's `fmm <C>, [vce(robust)]: nbreg <depvar> <regs>` — a C-component
NB2 finite mixture with constant class probabilities. Each component
has its own dispersion α_c:

    f(y|x) = Σ_c π_c · NB2(y; μ_c, α_c),  μ_c = exp(x'β_c),

class shares π_c via a reference-class logit (class 1 base). Jointly
MLE in (β_1..β_C, lnα_1..lnα_C, γ_2..γ_C) by NelderMead (gamma-safe,
no autodiff). Returns per-component β / α / μ and class shares for
downstream fitted-value work.

Returns `(; β_components, α, π, μ_components, γ, ll, V, se, n, k,
coefnames, β_pois_start, y, X)`.
"""
function stata_fmm_nbreg(df, formula; ncomponents::Int = 2,
                        level::Float64 = 0.95, quiet::Bool = false,
                        seed_split::Float64 = 0.1)
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
    βp = GLM.coef(m_pois); cn = GLM.coefnames(m_pois)
    y  = Float64.(GLM.response(m_pois)); X = GLM.modelmatrix(m_pois)
    n, k = size(X)

    # θ = [β_1..β_C (each k), lnα_1..lnα_C, γ_2..γ_C]
    off_α = C*k
    off_γ = C*k + C
    function _unpack(θ)
        B = reshape(view(θ, 1:C*k), k, C)
        α = exp.(view(θ, (off_α+1):(off_α+C)))
        γ = vcat(0.0, θ[(off_γ+1):end]); w = exp.(γ); π = w ./ sum(w)
        return B, α, π
    end
    # log NB2 component density (mixing weight folded in) for obs i, class c.
    _lnbcomp(yi, ηic, αc, lπc) = begin
        μ = exp(ηic); iα = 1/αc
        lπc + _c17_loggamma(yi + iα) -
              _c17_loggamma(iα) -
              _c17_loggamma(yi + 1) +
              iα*log(iα/(μ+iα)) + yi*log(μ/(μ+iα))
    end
    function nbfmm_nll(θ)
        B, α, π = _unpack(θ); Eta = X * B; lπ = log.(π)
        ll = zero(eltype(θ))
        # log-sum-exp over components (see note in stata_fmm_poisson —
        # exp-then-sum underflows for the long count tail).
        for i in 1:n
            yi = y[i]
            lc = _lnbcomp(yi, Eta[i,1], α[1], lπ[1])
            for c in 2:C
                lcc = _lnbcomp(yi, Eta[i,c], α[c], lπ[c])
                m   = max(lc, lcc); lc = m + log(exp(lc-m) + exp(lcc-m))
            end
            ll += -lc
        end
        return ll
    end

    θ0 = Float64[]
    for c in 1:C
        push!(θ0, (βp .* (1 + seed_split*(2*(c-1)/max(C-1,1) - 1)))...)
    end
    append!(θ0, zeros(C))        # lnα_c = 0 (α = 1)
    append!(θ0, zeros(C-1))      # γ_c = 0 (equal shares)
    res = Optim.optimize(nbfmm_nll, θ0, Optim.NelderMead(),
                         Optim.Options(iterations = 8000, g_tol = 1e-8))
    θ̂  = Float64.(Optim.minimizer(res)); ll = -nbfmm_nll(θ̂)::Float64

    # Deterministic component order: sort by ascending fitted mean
    # (matches Stata's class-1 = lower-mean convention). Reorder β and
    # lnα together and rebuild the class logits relative to the new base.
    let
        B0, α0, π0 = _unpack(θ̂)
        means = [Statistics.mean(exp.(X * B0[:, c])) for c in 1:C]
        ord   = sortperm(means)
        if ord != collect(1:C)
            Bs   = B0[:, ord]; lnαs = log.(α0[ord]); πs = π0[ord]
            γs   = [log(πs[c] / πs[1]) for c in 2:C]
            θ̂    = vcat(vec(Bs), lnαs, γs)
        end
    end
    ll = -nbfmm_nll(θ̂)::Float64
    B̂, α̂, π̂ = _unpack(θ̂)
    β_components = [collect(B̂[:, c]) for c in 1:C]
    μ_components = [exp.(X * β_components[c]) for c in 1:C]
    γ̂ = θ̂[(off_γ+1):end]
    nθ = length(θ̂)

    # Robust (Huber–White) vcov: A⁻¹ B A⁻¹ · n/(n−1), A = OIM Hessian,
    # B = Σ_i s_i s_iᵀ. Both via central finite differences (the NB
    # gamma terms have no ForwardDiff method here).
    function _fd_hess(f, x)
        np = length(x); H = zeros(np, np)
        h  = sqrt(sqrt(eps(Float64))) .* max.(abs.(x), 1.0); f0 = f(x)
        for i in 1:np
            xpi=copy(x);xmi=copy(x);xpi[i]+=h[i];xmi[i]-=h[i]
            H[i,i]=(f(xpi)-2*f0+f(xmi))/h[i]^2
            for j in (i+1):np
                a=copy(x);a[i]+=h[i];a[j]+=h[j]; b=copy(x);b[i]+=h[i];b[j]-=h[j]
                cc=copy(x);cc[i]-=h[i];cc[j]+=h[j]; d=copy(x);d[i]-=h[i];d[j]-=h[j]
                H[i,j]=H[j,i]=(f(a)-f(b)-f(cc)+f(d))/(4*h[i]*h[j])
            end
        end
        H
    end
    # single-obs nll for scores
    function nbfmm_nll_i(θ, i)
        B,α,π = _unpack(θ); lπ=log.(π); yi=y[i]
        lc = _lnbcomp(yi, LinearAlgebra.dot(view(X,i,:),view(B,:,1)), α[1], lπ[1])
        for c in 2:C
            lcc=_lnbcomp(yi, LinearAlgebra.dot(view(X,i,:),view(B,:,c)), α[c], lπ[c])
            m=max(lc,lcc); lc=m+log(exp(lc-m)+exp(lcc-m))
        end
        return -lc
    end
    V_oim = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hess(nbfmm_nll, θ̂)))
    hg = sqrt(sqrt(eps(Float64))) .* max.(abs.(θ̂), 1.0)
    S  = zeros(n, nθ)
    for i in 1:n, j in 1:nθ
        xp=copy(θ̂);xp[j]+=hg[j]; xm=copy(θ̂);xm[j]-=hg[j]
        S[i,j] = (nbfmm_nll_i(xp,i) - nbfmm_nll_i(xm,i)) / (2*hg[j])
    end
    V  = V_oim * (S'*S) * V_oim * (n/(n-1))
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))
    # parameter index helpers
    βidx(c, j) = (c-1)*k + j           # β_c[j]
    lnαidx(c)  = off_α + c             # lnα_c
    γidx(c)    = off_γ + (c-1)         # γ_c (c ≥ 2)

    if !quiet
        crit = Distributions.quantile(Distributions.Normal(), 1-(1-level)/2)
        function g9(x; w::Int=10, sig::Int=7)
            (ismissing(x) || !isfinite(x)) && return lpad(".", w)
            su=sig; s=Printf.@sprintf("%.*g", su, x)
            cap=(0<abs(x)<1 && x<0) ? 10 : 9
            while length(s)>cap && su>1; su-=1; s=Printf.@sprintf("%.*g",su,x); end
            0<abs(x)<1 && (s=replace(s, r"^(-?)0\."=>s"\1.")); lpad(s,w)
        end
        commafmt(num)=begin s=string(abs(num)); p=String[]; i=length(s)
            while i>=1; push!(p,s[max(1,i-2):i]); i-=3; end; (num<0 ? "-" : "")*join(reverse(p),","); end
        depname = string(formula.lhs)
        _row(lab,b,s) = begin z=b/s; pp=2*(1-Distributions.cdf(Distributions.Normal(),abs(z)))
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n", lab, g9(b;w=10), g9(s;w=9),
                Printf.@sprintf("%7.2f",z), Printf.@sprintf("%.3f",pp), g9(b-crit*s;w=9), g9(b+crit*s;w=10)); end
        _row_anc(lab,b,s) = Printf.@printf("%12s | %s  %s%26s%s  %s\n", lab, g9(b;w=10), g9(s;w=9), "",
                g9(b-crit*s;w=9), g9(b+crit*s;w=10))   # ancillary: no z / P>|z|
        ci = findfirst(==("(Intercept)"), cn)
        sl = setdiff(1:k, ci === nothing ? Int[] : [ci])
        ord_x = vcat(sl, ci === nothing ? Int[] : [ci])

        println()
        Printf.@printf("%-56s%-13s = %6s\n",
                       "Finite mixture model", "Number of obs", commafmt(n))
        Printf.@printf("Log pseudolikelihood = %.4f\n\n", ll)
        # latent-class logit block
        println("-"^78)
        Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [%g%% conf. interval]\n",
                       "", 100*level)
        println("-"^13, "+", "-"^64)
        Printf.@printf("%-12s | (base outcome)\n", "1.Class")
        for c in 2:C
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", "$(c).Class")
            _row("_cons", θ̂[γidx(c)], se[γidx(c)])
        end
        println("-"^78)
        # per-class nbreg blocks
        for c in 1:C
            println()
            Printf.@printf("Class       : %d\n", c)
            Printf.@printf("Response    : %s\n", depname)
            println("Model       : nbreg, dispersion(mean)")
            println()
            println("-"^78)
            Printf.@printf("%12s | Coefficient  std. err.      z    P>|z|     [%g%% conf. interval]\n",
                           "", 100*level)
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", depname)
            for j in ord_x
                lab = cn[j] == "(Intercept)" ? "_cons" : cn[j]
                _row(lab, B̂[j,c], se[βidx(c,j)])
            end
            println("-"^13, "+", "-"^64)
            Printf.@printf("%-12s |\n", "/$depname")
            _row_anc("lnalpha", log(α̂[c]), se[lnαidx(c)])
            println("-"^78)
        end
    end

    return (; β_components, α = α̂, π = π̂, μ_components, γ = γ̂, ll, V, se,
              n, k,
              coefnames = [x == "(Intercept)" ? "_cons" : x for x in cn],
              β_pois_start = βp, y, X)
end
