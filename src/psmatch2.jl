# (deps provided by the StatEcon module)

# Reproduces Stata's `psmatch2 treat covars, outcome(y) logit ate`.
#
#   psmatch2(df, :train,
#            [:age, :agesq, :educ, :educsq, :age_educ, :unem96,
#             :earn96, :earn96sq, :age_earn96, :educ_earn96];
#            outcome = :earn98)
#
# Steps, matching psmatch2's defaults:
#   1. Estimate the propensity score with a logit of `treat` on `covars`
#      (option `logit`; psmatch2's default link is also logit).
#   2. Single nearest-neighbour matching ON THE PROPENSITY SCORE, WITH
#      REPLACEMENT (psmatch2's default n(1), with replacement). Ties are
#      resolved by averaging all equally-closest neighbours (equal weights).
#   3. Report ATT, ATU and ATE (option `ate`).
#
# Standard errors use psmatch2's own "fixed-weights" analytic formula
#   Var = s1^2/N1 + (sum w_j^2) s0^2 / N1^2      (and symmetrically for ATU),
# where w_j counts how often a control is used and s1^2, s0^2 are the
# within-group outcome variances. Abadie & Imbens (2006) show this SE is not
# fully consistent; psmatch2 reports it nonetheless, so it is reproduced here.
# `ratio` adds ATT+ATU weighting for ATE. Pass `psscore` to reuse an existing
# score instead of re-estimating.

function psmatch2(df::AbstractDataFrame, treat::Symbol, covars::AbstractVector{Symbol};
                  outcome::Symbol, psscore::Union{Nothing,Symbol}=nothing,
                  tol::Float64=1e-8)

    # ---- 1. propensity score ------------------------------------------
    keep = completecases(df[:, unique([treat; outcome; covars])])   # e(sample)
    work = df[keep, :]
    y = Float64.(work[!, outcome])
    t = Int.(work[!, treat])

    if isnothing(psscore)
        fm = Term(treat) ~ sum(Term.(covars))
        m  = GLM.glm(fm, work, Binomial(), LogitLink())
        ps = GLM.predict(m)
    else
        ps = Float64.(work[!, psscore])
    end

    tidx = findall(==(1), t)          # treated rows
    cidx = findall(==(0), t)          # control rows
    N1, N0 = length(tidx), length(cidx)

    # nearest neighbour(s) of `i` within candidate set `cand` (on ps)
    function nn(i, cand)
        d = abs.(ps[cand] .- ps[i])
        dmin = minimum(d)
        cand[findall(x -> x <= dmin + tol, d)]     # all ties
    end

    # ---- 2a. ATT: match each treated to nearest control ----------------
    att_i = Vector{Float64}(undef, N1)
    wctrl = Dict{Int,Float64}()                    # control-usage weights
    for (k, i) in enumerate(tidx)
        ms = nn(i, cidx)
        att_i[k] = y[i] - Statistics.mean(y[ms])
        for j in ms
            wctrl[j] = get(wctrl, j, 0.0) + 1 / length(ms)
        end
    end
    ATT = Statistics.mean(att_i)

    # ---- 2b. ATU: match each control to nearest treated ----------------
    atu_j = Vector{Float64}(undef, N0)
    wtreat = Dict{Int,Float64}()
    for (k, j) in enumerate(cidx)
        ms = nn(j, tidx)
        atu_j[k] = Statistics.mean(y[ms]) - y[j]
        for i in ms
            wtreat[i] = get(wtreat, i, 0.0) + 1 / length(ms)
        end
    end
    ATU = Statistics.mean(atu_j)

    # ---- 3. ATE --------------------------------------------------------
    ATE = (N1 * ATT + N0 * ATU) / (N1 + N0)

    # Write _pscore and _weight back into the caller's data frame, as Stata's
    # psmatch2 does, so companion commands (e.g. psgraph) can use them.
    ps_full = Vector{Union{Missing,Float64}}(missing, DataFrames.nrow(df))
    ps_full[keep] = ps
    df[!, :_pscore] = ps_full
    w_full = Vector{Union{Missing,Float64}}(missing, DataFrames.nrow(df))
    keeprows = findall(keep)                       # map work-index -> df-row
    for i in tidx;              w_full[keeprows[i]] = 1.0;          end   # treated
    for (j, w) in wctrl;        w_full[keeprows[j]] = w;            end   # matched controls
    df[!, :_weight] = w_full

    # ---- psmatch2-style fixed-weights variances ------------------------
    s1sq = Statistics.var(y[tidx]); s0sq = Statistics.var(y[cidx])
    sumw_ctrl  = sum(w -> w^2, values(wctrl))
    sumw_treat = sum(w -> w^2, values(wtreat))
    var_att = s1sq / N1 + sumw_ctrl  * s0sq / N1^2
    var_atu = s0sq / N0 + sumw_treat * s1sq / N0^2
    var_ate = (N1^2 * var_att + N0^2 * var_atu) / (N1 + N0)^2
    se = sqrt.([var_att, var_atu, var_ate])

    # treated / control mean levels shown by psmatch2
    treated_mean = [Statistics.mean(y[tidx]), Statistics.mean(y[cidx] .+ atu_j), Statistics.mean(y)]
    ctrl_mean    = treated_mean .- [ATT, ATU, ATE]
    effects      = [ATT, ATU, ATE]
    tstat        = effects ./ se

    # ---- print psmatch2-style table ------------------------------------
    Printf.@printf("%-8s %-9s | %10s %10s %11s %11s %9s\n",
            "Variable", "Sample", "Treated", "Controls", "Difference", "S.E.", "T-stat")
    println("-"^76)
    labels = ["ATT", "ATU", "ATE"]
    for r in 1:3
        v = r == 1 ? string(outcome) : ""
        Printf.@printf("%-8s %-9s | %10.5g %10.5g %11.5g %11.5g %9.2f\n",
                v, labels[r], treated_mean[r], ctrl_mean[r], effects[r], se[r], tstat[r])
    end

    return DataFrames.DataFrame(Sample = labels, Treated = treated_mean, Controls = ctrl_mean,
                     Difference = effects, SE = se, T = tstat)
end
