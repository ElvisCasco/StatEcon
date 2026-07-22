# ============================================================================
# stata_zinb.jl — Stata `zinb` zero-inflated NB2 (Cameron & Trivedi ch17)
#
# Jointly MLE by Optim.NelderMead (derivative-free; the NB2 log-gamma terms use
# the ch17-shared `_c17_loggamma` from stata_ztnb.jl). The optional Vuong test
# fits a plain NB2 via `stata_nbreg` and reads its `.β_glm` / `.lnalpha` / `.ll`.
# ============================================================================

import Optim

"""
    stata_zinb(df, count_formula; inflate_vars, vce=:oim, level=0.95,
               quiet=false, vuong=true) -> NamedTuple

Stata's `zinb <y> <xc>, inflate(<xi>) [forcevuong] nolog` — zero-inflated
NB2. Two equations: the NB2 count mean μ = exp(x'β) (dispersion α), and a
logit "always-zero" inflation π = Λ(z'γ). Density

    P(0) = π + (1−π)·(1+α·μ)^{−1/α}
    P(y) = (1−π)·NB2(y; μ, α),   y > 0.

Jointly MLE in (β, γ, lnα) by NelderMead (the NB gamma terms are handled
by `_c17_loggamma`), OIM vcov from a central-difference Hessian (robust
sandwich with `vce=:robust`). Prints the Stata `zinb` block — count
equation, `inflate` equation, `/lnalpha` + `alpha` rows — and, when
`vuong=true`, the Vuong test of ZINB vs plain NB2 (`forcevuong`).

Returns `(; β, γ, lnα, α, V, se, ll, ll_null, ll_nb, vuong_z, vuong_p, n,
n_zero, coefnames_count, coefnames_inflate)`.
"""
function stata_zinb(df, count_formula; inflate_vars::AbstractVector{Symbol},
                   vce::Symbol = :oim, level::Float64 = 0.95,
                   quiet::Bool = false, vuong::Bool = true)
    needed = unique(vcat(StatsModels.termvars(count_formula),
                         collect(inflate_vars)))
    dfc = DataFrames.dropmissing(df[:, needed])
    for c in needed
        col = dfc[!, c]
        if eltype(col) <: Union{Missing,Float32} || eltype(col) === Float32
            dfc[!, c] = Float64.(col)
        end
    end
    m_pois = GLM.glm(count_formula, dfc, Distributions.Poisson(), GLM.LogLink())
    βp  = GLM.coef(m_pois); cn_x = GLM.coefnames(m_pois)
    y   = Float64.(GLM.response(m_pois)); X = GLM.modelmatrix(m_pois)
    n, kx = size(X)
    Z   = hcat([Float64.(_c17_rawval.(dfc[!, v])) for v in inflate_vars]...,
               ones(n))
    cn_z = vcat(string.(inflate_vars), "_cons")
    kz  = size(Z, 2)
    nz  = count(==(0), y)

    # NB2 log-pmf pieces.
    _lnnb(yi, μi, iα) = _c17_loggamma(yi + iα) -
                        _c17_loggamma(iα) -
                        _c17_loggamma(yi + 1) +
                        iα*log(iα/(iα + μi)) + yi*log(μi/(iα + μi))

    function nll(θ)
        β = θ[1:kx]; γ = θ[(kx+1):(kx+kz)]; α = exp(θ[end]); iα = 1/α
        μ = exp.(X*β); ηi = Z*γ
        ll = zero(eltype(θ))
        for i in 1:n
            πi = 1/(1 + exp(-ηi[i])); μi = μ[i]
            if y[i] == 0
                p0nb = (iα/(iα + μi))^iα
                ll += log(πi + (1 - πi)*p0nb)
            else
                ll += log1p(-πi) + _lnnb(y[i], μi, iα)
            end
        end
        return -ll
    end

    θ0  = Float64.(vcat(βp, zeros(kz), 0.0))
    res = Optim.optimize(nll, θ0, Optim.NelderMead(),
                         Optim.Options(iterations = 8000, g_tol = 1e-8))
    θ̂  = Float64.(Optim.minimizer(res)); ll = -nll(θ̂)::Float64
    β̂  = θ̂[1:kx]; γ̂ = θ̂[(kx+1):(kx+kz)]; lnα = θ̂[end]; α = exp(lnα)

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
    nθ = length(θ̂)
    V_oim = LinearAlgebra.inv(LinearAlgebra.Symmetric(_fd_hess(nll, θ̂)))
    V = if vce == :robust
        function nll_i(θ, i)
            β=θ[1:kx]; γ=θ[(kx+1):(kx+kz)]; α_=exp(θ[end]); iα=1/α_
            μi=exp(LinearAlgebra.dot(view(X,i,:),β)); πi=1/(1+exp(-LinearAlgebra.dot(view(Z,i,:),γ)))
            if y[i]==0
                return -log(πi + (1-πi)*(iα/(iα+μi))^iα)
            else
                return -(log1p(-πi) + _lnnb(y[i], μi, iα))
            end
        end
        hg = sqrt(sqrt(eps(Float64))) .* max.(abs.(θ̂), 1.0)
        S  = zeros(n, nθ)
        for i in 1:n, j in 1:nθ
            xp=copy(θ̂);xp[j]+=hg[j]; xm=copy(θ̂);xm[j]-=hg[j]
            S[i,j]=(nll_i(xp,i)-nll_i(xm,i))/(2*hg[j])
        end
        V_oim*(S'*S)*V_oim*(n/(n-1))
    else
        V_oim
    end
    se = sqrt.(max.(LinearAlgebra.diag(V), 0.0))

    # Vuong test: ZINB vs plain NB2 (per-obs log-density difference).
    vuong_z = NaN; vuong_p = NaN; ll_nb = NaN
    if vuong
        m_nb = stata_nbreg(dfc, count_formula; vce = :oim, quiet = true)
        β_nb = m_nb.β_glm; lnα_nb = m_nb.lnalpha; iα_nb = 1/exp(lnα_nb)
        μ_nb = exp.(X * β_nb)
        μ_z  = exp.(X * β̂); ηi = Z*γ̂; iα = 1/α
        mvec = zeros(n)
        for i in 1:n
            πi = 1/(1 + exp(-ηi[i]))
            f_zinb = y[i]==0 ? πi + (1-πi)*(iα/(iα+μ_z[i]))^iα :
                               (1-πi)*exp(_lnnb(y[i], μ_z[i], iα))
            f_nb   = exp(_lnnb(y[i], μ_nb[i], iα_nb))
            mvec[i] = log(max(f_zinb, eps())) - log(max(f_nb, eps()))
        end
        ll_nb = m_nb.ll
        vuong_z = sqrt(n) * Statistics.mean(mvec) / Statistics.std(mvec)
        vuong_p = 1 - Distributions.cdf(Distributions.Normal(), vuong_z)
    end

    # ll(null) = Stata's `e(ll_0)`: the base for the count-equation LR
    # test — count equation reduced to its intercept, inflation
    # equation kept as specified. (NOT intercept-only everywhere, which
    # collapses to the plain-NB null.)
    ll_null = let
        function nll0(θ)
            b0 = θ[1]; γ = view(θ, 2:(1+kz)); α_ = exp(θ[end]); iα = 1/α_
            μ0 = exp(b0); ηi = Z*γ; s = zero(eltype(θ))
            for i in 1:n
                πi = 1/(1 + exp(-ηi[i]))
                if y[i] == 0
                    s += log(πi + (1-πi)*(iα/(iα+μ0))^iα)
                else
                    s += log1p(-πi) + _lnnb(y[i], μ0, iα)
                end
            end
            return -s
        end
        ypos = y[y .> 0]
        θ0n = vcat(log(isempty(ypos) ? 1.0 : Statistics.mean(ypos)),
                   zeros(kz), 0.0)
        rn = Optim.optimize(nll0, θ0n, Optim.NelderMead(),
                            Optim.Options(iterations = 8000, g_tol = 1e-9))
        -Optim.minimum(rn)
    end

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
        depname = string(count_formula.lhs)
        ll_lab  = vce==:robust ? "Log pseudolikelihood" : "Log likelihood"
        _row(lab,b,s)=begin z=b/s; pp=2*(1-Distributions.cdf(Distributions.Normal(),abs(z)))
            Printf.@printf("%12s | %s  %s  %s   %s    %s  %s\n", lab, g9(b;w=10), g9(s;w=9),
                Printf.@sprintf("%7.2f",z), Printf.@sprintf("%.3f",pp), g9(b-crit*s;w=9), g9(b+crit*s;w=10)); end
        _row_anc(lab,b,s)=Printf.@printf("%12s | %s  %s%26s%s  %s\n", lab, g9(b;w=10), g9(s;w=9), "",
                g9(b-crit*s;w=9), g9(b+crit*s;w=10))
        cix = findfirst(==("(Intercept)"), cn_x)
        slx = setdiff(1:kx, cix === nothing ? Int[] : [cix])
        ordx = vcat(slx, cix === nothing ? Int[] : [cix])

        println()
        Printf.@printf("%-52s%-13s = %8s\n",
                       "Zero-inflated negative binomial regression",
                       "Number of obs", commafmt(n))
        Printf.@printf("%52s%-13s = %8s\n", "", "Nonzero obs", commafmt(n-nz))
        Printf.@printf("%52s%-13s = %8s\n", "", "Zero obs", commafmt(nz))
        Printf.@printf("Inflation model = logit\n")
        Printf.@printf("%s = %.4f\n\n", ll_lab, ll)
        se_lab = vce==:robust ? "std. err." : "Std. err."
        println("-"^78)
        Printf.@printf("%12s | Coefficient  %s      z    P>|z|     [%g%% conf. interval]\n",
                       depname, se_lab, 100*level)
        println("-"^13, "+", "-"^64)
        # count equation
        Printf.@printf("%-12s |\n", depname)
        for j in ordx
            lab = cn_x[j] == "(Intercept)" ? "_cons" : cn_x[j]
            _row(lab, β̂[j], se[j])
        end
        println("-"^13, "+", "-"^64)
        # inflate equation
        Printf.@printf("%-12s |\n", "inflate")
        ciz = findfirst(==("_cons"), cn_z)
        slz = setdiff(1:kz, ciz === nothing ? Int[] : [ciz])
        for j in vcat(slz, ciz === nothing ? Int[] : [ciz])
            _row(cn_z[j], γ̂[j], se[kx + j])
        end
        println("-"^13, "+", "-"^64)
        _row_anc("/lnalpha", lnα, se[end])
        println("-"^13, "+", "-"^64)
        _row_anc("alpha", α, α*se[end])      # delta method on exp(lnα)
        println("-"^78)
        if vuong && isfinite(vuong_z)
            Printf.@printf("Vuong test of zinb vs. standard negbin: z = %5.2f  Pr>z = %.4f\n",
                           vuong_z, vuong_p)
        end
    end

    return (; β = β̂, γ = γ̂, lnα, α, V, se, ll, ll_null, ll_nb,
              vuong_z, vuong_p, n, n_zero = nz,
              coefnames_count = [x=="(Intercept)" ? "_cons" : x for x in cn_x],
              coefnames_inflate = cn_z)
end
