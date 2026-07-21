# (deps provided by the Stat_Econ module)

# Reproduces Stata's `summarize varlist [if]`.
#
#   stata_summarize(df, :ph)                       # summarize ph
#   stata_summarize(df, [:ph, :age])               # summarize ph age
#   stata_summarize(df, :ph; if_ = df.train .== 1) # sum ph if train
#
# Prints the Obs / Mean / Std. dev. / Min / Max table and returns the
# summary as a DataFrame. Missing values are dropped per variable, as Stata does.

# Stata prints numbers like .5326870 (leading zero suppressed, 7 significant digits).
function _stata_num(x::Real)
    isnan(x) && return "."
    s = @sprintf("%.7g", x)
    s = replace(s, r"^(-?)0\." => s"\g<1>.")   # 0.53 -> .53 ,  -0.02 -> -.02
    return s
end

function stata_summarize(df::AbstractDataFrame, cols;
                         if_::Union{Nothing,AbstractVector{<:Union{Bool,Missing}}}=nothing)
    colsyms = cols isa Symbol ? [cols] : collect(cols)
    sub = isnothing(if_) ? df : df[coalesce.(if_, false), :]

    names_ = String[]; obs = Int[]; mean_ = Float64[]
    sd = Float64[]; mn = Float64[]; mx = Float64[]
    for c in colsyms
        v = collect(skipmissing(sub[!, c]))
        push!(names_, string(c)); push!(obs, length(v))
        if isempty(v)
            push!(mean_, NaN); push!(sd, NaN); push!(mn, NaN); push!(mx, NaN)
        else
            push!(mean_, mean(v)); push!(sd, std(v))
            push!(mn, minimum(v)); push!(mx, maximum(v))
        end
    end

    # ---- column widths -------------------------------------------------
    varw = max(maximum(length.(names_)), length("Variable"))
    obsw = max(maximum(length.(string.(obs))), length("Obs"), 3)
    meanstrs = _stata_num.(mean_); sdstrs = _stata_num.(sd)
    mnstrs = _stata_num.(mn); mxstrs = _stata_num.(mx)
    numw = maximum(length.(vcat(meanstrs, sdstrs, mnstrs, mxstrs,
                                "Mean", "Std. dev.", "Min", "Max")))

    pad(s, w) = lpad(s, w)
    header = pad("Variable", varw) * " | " *
             pad("Obs", obsw) * "  " * pad("Mean", numw) * "  " *
             pad("Std. dev.", numw) * "  " * pad("Min", numw) * "  " *
             pad("Max", numw)
    println(header)
    println("-"^(varw + 1) * "+" * "-"^(length(header) - varw - 2))
    for i in eachindex(names_)
        println(pad(names_[i], varw) * " | " *
                pad(string(obs[i]), obsw) * "  " *
                pad(meanstrs[i], numw) * "  " * pad(sdstrs[i], numw) * "  " *
                pad(mnstrs[i], numw) * "  " * pad(mxstrs[i], numw))
    end

    return DataFrame(Variable = names_, Obs = obs, Mean = mean_,
                     Std = sd, Min = mn, Max = mx)
end
