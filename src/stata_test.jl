"""
    stata_test(model, vars; values=zeros(length(vars)))

Stata-style `test <varlist>` after a regression — joint Wald test of
`H0: βⱼ = value[j]` for each `j` in `vars`. Uses the model's vcov (so it
respects robust/cluster SE if fitted that way). Reports the F statistic
`F(q, dof_resid)` and p-value, matching Stata's `test` after `regress`.

`vars` may be strings, symbols, or mixed. Coefficients with non-finite β
or vcov row are dropped (mirroring Stata's handling of collinear terms).
Returns `(F, df1, df2, p, chi2)`.
"""
function stata_test(model, vars::AbstractVector; values = zeros(length(vars)))
    β     = StatsBase.coef(model)
    V     = StatsBase.vcov(model)
    names = string.(StatsBase.coefnames(model))
    vs    = string.(vars)
    idx   = [findfirst(==(v), names) for v in vs]
    miss  = findall(x -> x === nothing, idx)
    isempty(miss) || error("variable(s) not found in model: $(vs[miss])")

    vals = collect(values)
    bad  = findall(k -> !isfinite(β[idx[k]]) ||
                        any(!isfinite, V[idx[k], idx]),
                   eachindex(idx))
    if !isempty(bad)
        @warn "Dropping collinear / non-finite coefficient(s): $(vs[bad])"
        keep = setdiff(eachindex(idx), bad)
        vs, idx, vals = vs[keep], idx[keep], vals[keep]
    end
    isempty(vs) && error("No testable coefficients remain.")

    q    = length(vs)
    Rβ   = β[idx] .- vals
    W    = LinearAlgebra.dot(Rβ, V[idx, idx] \ Rβ)
    dfr  = StatsBase.dof_residual(model)
    Fst  = W / q
    p    = 1 - Distributions.cdf(Distributions.FDist(q, dfr), Fst)

    println("Wald F test (respects model's vcov):")
    for (k, v) in enumerate(vs)
        Printf.@printf("  ( %d)  %s = %g\n", k, v, vals[k])
    end
    Printf.@printf("\n       F( %d, %g) = %.4f\n", q, dfr, Fst)
    Printf.@printf("            Prob > F = %.4f\n", p)
    return (F = Fst, df1 = q, df2 = dfr, p = p, chi2 = W)
end
