# (deps provided by the StatEcon module)

"""
    stata_predict_heckman(df, fit; outcome_regs=fit.outcome_regs,
                          selection_regs=fit.selection_regs,
                          textbook_bug=true) -> NamedTuple

Stata `heckman` post-estimation predictions. Given a [`stata_heckman`](@ref)
fit and a DataFrame `df`, returns a NamedTuple of vectors

  `probpos`     = Φ(z'γ̂)                                (Stata `predict, psel`)
  `x1b1`        = z'γ̂                                    (Stata `predict, xbsel`)
  `x2b2`        = x'β̂                                    (Stata `predict, xb`)
  `sig2sq`      = σ̂²                                     (`e(sigma)^2`)
  `sig12sq`     = ρ̂·σ̂²  (textbook_bug=true)              matches the textbook
                  ρ̂·σ̂    (textbook_bug=false)             mathematically correct
  `yhatheck`    = exp(x2b2 + ½σ²)·(1 − Φ(−x1b1 − sig12sq))   (E[y|x])
  `yhatposheck` = yhatheck / probpos                          (E[y|x, dy=1])

The default `textbook_bug=true` reproduces Cameron & Trivedi's Stata recipe
(`scalar sig12sq = e(rho)*e(sigma)^2`), whose σ² ought to be σ for the
lognormal correction to be exact.
"""
function stata_predict_heckman(df::DataFrames.AbstractDataFrame, fit;
                               outcome_regs::AbstractVector{Symbol} =
                                   Symbol.(fit.outcome_regs),
                               selection_regs::AbstractVector{Symbol} =
                                   Symbol.(fit.selection_regs),
                               textbook_bug::Bool = true)
    N    = DataFrames.nrow(df)
    Xmat = hcat([Float64.(_c16_rawval.(df[!, r])) for r in outcome_regs]...,
                ones(N))
    Zmat = hcat([Float64.(_c16_rawval.(df[!, r])) for r in selection_regs]...,
                ones(N))
    Φ_(z) = _c16_Phi(z)

    x1b1     = Zmat * fit.γ
    x2b2     = Xmat * fit.β
    probpos  = Φ_.(x1b1)
    sig2sq   = fit.σ^2
    sig12sq  = textbook_bug ? fit.ρ * fit.σ^2 : fit.ρ * fit.σ

    yhatheck    = exp.(x2b2 .+ 0.5 .* sig2sq) .*
                  (1 .- Φ_.(.-x1b1 .- sig12sq))
    yhatposheck = yhatheck ./ probpos

    return (; probpos, x1b1, x2b2,
              sig2sq, sig12sq, sigma1sq = 1.0,
              yhatheck, yhatposheck)
end
