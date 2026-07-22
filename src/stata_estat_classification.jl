# --------------------------------------------------------------------------
# Cameron & Trivedi ch. 14 — Binary outcome models
#   stata_estat_classification — `estat classification` after a binary model
# --------------------------------------------------------------------------

"""
    stata_estat_classification(m, df; depvar="", cutoff=0.5) -> NamedTuple

Stata-style `estat classification` after a binary model. Prints the full
Stata classification block:

  - Confusion matrix (Classified +/- × True D/~D × Total)
  - "Classified + if predicted Pr(D) >= <cutoff>" / "True D defined as
    <depvar> != 0"
  - Sensitivity / Specificity / PPV / NPV
  - False+/− rates for true ~D / D / classified +/−
  - Correctly classified

`m` is a fitted GLM binary model (e.g. `stata_logit(...).model`) and `df`
the estimation sample. `depvar` is the dependent-variable name shown in the
headers (e.g. `"ins"`); if empty, falls back to the first coefname.
"""
function stata_estat_classification(m, df; depvar::AbstractString = "",
                                    cutoff::Float64 = 0.5)
    phat = GLM.predict(m, df)
    yhat = Int.(phat .>= cutoff)
    y    = Int.(GLM.response(m))
    tp = sum((yhat .== 1) .& (y .== 1))
    tn = sum((yhat .== 0) .& (y .== 0))
    fp = sum((yhat .== 1) .& (y .== 0))
    fn = sum((yhat .== 0) .& (y .== 1))
    n  = length(y)

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv  = tp / (tp + fp)
    npv  = tn / (tn + fn)
    fp_rate_negD = fp / (tn + fp)
    fn_rate_posD = fn / (tp + fn)
    fp_rate_clsP = fp / (tp + fp)
    fn_rate_clsN = fn / (tn + fn)
    correct = (tp + tn) / n

    dv = isempty(depvar) ? string(StatsBase.coefnames(m)[1]) : depvar
    sep_box = "-"^11 * "+" * "-"^26 * "+" * "-"^11
    sep_pct = "-"^50

    println("Logistic model for $dv")
    println()
    println("              -------- True --------")
    println("Classified |         D            ~D  |      Total")
    println(sep_box)
    Printf.@printf("     +     |    %6d        %6d  |    %7d\n",
                   tp, fp, tp + fp)
    Printf.@printf("     -     |    %6d        %6d  |    %7d\n",
                   fn, tn, fn + tn)
    println(sep_box)
    Printf.@printf("   Total   |    %6d        %6d  |    %7d\n",
                   tp + fn, tn + fp, n)
    println()
    Printf.@printf("Classified + if predicted Pr(D) >= %s\n",
                   replace(Printf.@sprintf("%.2f", cutoff),
                           r"^0\." => "."))
    Printf.@printf("True D defined as %s != 0\n", dv)
    println(sep_pct)
    Printf.@printf("%-32sPr( +| D) %7.2f%%\n", "Sensitivity",                 100*sens)
    Printf.@printf("%-32sPr( -|~D) %7.2f%%\n", "Specificity",                 100*spec)
    Printf.@printf("%-32sPr( D| +) %7.2f%%\n", "Positive predictive value",   100*ppv)
    Printf.@printf("%-32sPr(~D| -) %7.2f%%\n", "Negative predictive value",   100*npv)
    println(sep_pct)
    Printf.@printf("%-32sPr( +|~D) %7.2f%%\n", "False + rate for true ~D",        100*fp_rate_negD)
    Printf.@printf("%-32sPr( -| D) %7.2f%%\n", "False - rate for true D",         100*fn_rate_posD)
    Printf.@printf("%-32sPr(~D| +) %7.2f%%\n", "False + rate for classified +",   100*fp_rate_clsP)
    Printf.@printf("%-32sPr( D| -) %7.2f%%\n", "False - rate for classified -",   100*fn_rate_clsN)
    println(sep_pct)
    Printf.@printf("%-42s%7.2f%%\n", "Correctly classified", 100*correct)
    println(sep_pct)

    return (; tp, tn, fp, fn,
              sensitivity = sens, specificity = spec, ppv, npv,
              fp_rate_negD, fn_rate_posD, fp_rate_clsP, fn_rate_clsN,
              accuracy = correct, n)
end
