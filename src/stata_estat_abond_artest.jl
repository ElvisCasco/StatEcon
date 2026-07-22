# ============================================================================
# stata_estat_abond_artest.jl — Stata panel-data extensions (Cameron & Trivedi ch09)
# ============================================================================

"""
    stata_estat_abond_artest(res, m::Int=1)

Arellano-Bond test for zero autocorrelation of order `m` in the
first-differenced residuals. Pass the NamedTuple returned by
`stata_arellano_bond` or `stata_xtdpdsys`. Returns `(; m, z, p)`.
"""
function stata_estat_abond_artest(res, m::Int=1)
    T_vals = hasproperty(res, :T_diff_per_panel) ? res.T_diff_per_panel : res.T_vals
    N_ind  = res.N_ind
    u_all  = res.residuals[1:res.N_obs]
    panels = Vector{Vector{Float64}}(undef, N_ind)
    offset = 0
    for i in 1:N_ind
        Ti = T_vals[i]
        panels[i] = u_all[offset+1:offset+Ti]
        offset += Ti
    end

    num = 0.0
    den = 0.0
    for i in 1:N_ind
        ei = panels[i]
        Ti = length(ei)
        Ti <= m && continue
        e_cur = ei[(m+1):Ti]
        e_lag = ei[1:(Ti-m)]
        cross = LinearAlgebra.dot(e_lag, e_cur)
        num  += cross
        den  += cross^2
    end
    z  = num / sqrt(den)
    pv = 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(z)))
    return (; m, z, p=pv)
end
