"""
    stata_estat_covariance(res; differenced=false) -> Matrix
    stata_estat_correlation(res; differenced=false) -> Matrix

Stata `estat covariance` / `estat correlation` after `asmprobit`.
For our independent-errors `stata_asmprobit`, the *structural* (J×J)
matrix is the identity (no free correlation parameters). The
*differenced* form (J-1)×(J-1) for iid is `Σ̃ = D·I·Dᵀ = [[2,1,…],
[1,2,…],…]` with off-diagonal correlation 0.5 (every differenced
pair shares the base alternative's latent error). Pass
`differenced = true` to print the differenced matrix.

Returns the matrix it prints. Note: this is the *implied* matrix for
an iid fit; not Stata's full `correlation(unstructured)` output.
"""
function stata_estat_covariance(res; differenced::Bool = false)
    alts = collect(res.alts); J = res.J
    if !differenced
        labels = alts
        Σ = Matrix{Float64}(LinearAlgebra.I, J, J)
    else
        labels = collect(res.nonbase)
        Σ = ones(J - 1, J - 1) .+
            Matrix{Float64}(LinearAlgebra.I, J - 1, J - 1)
    end
    title = differenced ? "Covariance of latent-error differences" :
                          "Structural covariance of latent errors"
    println(); println(title)
    println("(independent-errors asmprobit: structural Σ = I; ",
            "differenced [[2,1,…],[1,2,…],…])")
    println()
    w = max(8, maximum(length, labels; init = 0) + 1)
    print(lpad("", w)); for l in labels; print(lpad(l, w)); end; println()
    println("-"^(w * (length(labels) + 1)))
    for (i, ri) in enumerate(labels)
        print(lpad(ri, w))
        for (j, _) in enumerate(labels)
            print(lpad(Printf.@sprintf("%.4f", Σ[i, j]), w))
        end
        println()
    end
    return Σ
end

"""
    stata_estat_correlation(res; differenced=false) -> Matrix

Stata `estat correlation` after `stata_asmprobit`. Correlation counterpart
of [`stata_estat_covariance`](@ref): the structural form is the identity;
the `differenced=true` form has 0.5 off-diagonals (each differenced pair
shares the base alternative's latent error). Returns the printed matrix.
"""
function stata_estat_correlation(res; differenced::Bool = false)
    Σ = differenced ?
        (ones(res.J - 1, res.J - 1) .+
         Matrix{Float64}(LinearAlgebra.I, res.J - 1, res.J - 1)) :
        Matrix{Float64}(LinearAlgebra.I, res.J, res.J)
    d = sqrt.(LinearAlgebra.diag(Σ))
    C = Σ ./ (d * d')
    labels = differenced ? collect(res.nonbase) : collect(res.alts)
    title = differenced ? "Correlations of latent-error differences" :
                          "Structural correlations of latent errors"
    println(); println(title)
    println("(independent-errors asmprobit: structural C = I; ",
            "differenced off-diag = 0.5)")
    println()
    w = max(8, maximum(length, labels; init = 0) + 1)
    print(lpad("", w)); for l in labels; print(lpad(l, w)); end; println()
    println("-"^(w * (length(labels) + 1)))
    for (i, ri) in enumerate(labels)
        print(lpad(ri, w))
        for (j, _) in enumerate(labels)
            print(lpad(Printf.@sprintf("%.4f", C[i, j]), w))
        end
        println()
    end
    return C
end
