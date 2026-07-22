# (deps provided by the StatEcon module)

# Reproduces Stata's `summarize varlist [if] [, detail]`.
#
#   stata_summarize(df)                            # summarize   (all numeric vars)
#   stata_summarize(df, :ph)                       # summarize ph
#   stata_summarize(df, [:ph, :age])               # summarize ph age
#   stata_summarize(df, "t*")                      # summarize t*   (glob varlist)
#   stata_summarize(df, "price-weight")            # summarize price-weight
#   stata_summarize(df, :ph; if_ = df.train .== 1) # summarize ph if train
#   stata_summarize(df, :price; detail = true)     # summarize price, detail
#
# Prints the Obs / Mean / Std. dev. / Min / Max table (or Stata's full
# `detail` block) and returns the base summary as a DataFrame. Missing values
# are dropped per variable, as Stata does. `ReadStatTables.LabeledValue`s are
# transparently unwrapped, so this works directly on `.dta` files.

# Unwrap ReadStatTables.LabeledValue -> its underlying numeric value.
# (Any type with a `.value` field is unwrapped; plain numbers pass through.)
_sm_rawval(x) = hasproperty(x, :value) ? x.value : x

# Convert `col` to a Vector{Float64} of non-missing values, unwrapping
# LabeledValue when needed. Returns `nothing` when the column is non-numeric
# (e.g. String), so the caller can skip it.
function _sm_numeric_vec(col)
    vals = collect(skipmissing(col))
    isempty(vals) && return Float64[]
    T = Base.nonmissingtype(eltype(col))
    T <: Real && return convert(Vector{Float64}, vals)
    try
        return Float64[Float64(_sm_rawval(v)) for v in vals]
    catch
        return nothing                   # non-numeric (strings, etc.)
    end
end

# Expand a Stata-style varlist pattern against `df`'s columns. Supports:
#   * `t*`, `*t`, `*mpg*`, `t?`   — glob wildcards
#   * `price-weight`              — range of columns (inclusive, in column order)
#   * exact names, Symbols, vectors, and the literal "_all"
function _sm_expand_varlist(df, vars)
    cols = string.(DataFrames.names(df))
    patterns = vars isa Union{Symbol,AbstractString} ? [string(vars)] :
               vars isa AbstractVector ? [string(v) for v in vars] :
               [string(v) for v in collect(vars)]

    out = String[]
    for p in patterns
        if p == "_all"
            append!(out, cols)
        elseif occursin('*', p) || occursin('?', p)
            rx = Regex("^" * replace(p, "*" => ".*", "?" => ".") * "\$")
            for c in cols
                occursin(rx, c) && push!(out, c)
            end
        elseif occursin('-', p) && !startswith(p, "-") && !(p in cols)
            parts = split(p, '-', limit = 2)
            i = findfirst(==(String(parts[1])), cols)
            j = findfirst(==(String(parts[2])), cols)
            (i === nothing || j === nothing) && error("Unknown variable in range '$p'")
            i, j = minmax(i, j)
            append!(out, cols[i:j])
        else
            push!(out, p)
        end
    end
    seen = Set{String}(); uniq = String[]
    for v in out
        v in seen || (push!(seen, v); push!(uniq, v))
    end
    return Symbol.(uniq)
end

# Stata prints numbers like .5326870 (leading zero suppressed, 7 significant digits).
function _stata_num(x::Real)
    isnan(x) && return "."
    s = @sprintf("%.7g", x)
    s = replace(s, r"^(-?)0\." => s"\g<1>.")   # 0.53 -> .53 ,  -0.02 -> -.02
    return s
end

# Stata %9.0g: up to `mx` sig figs, capped at `w` chars, leading "0" stripped for |x|<1.
function _sm_g9(x; w::Int = 9, mx::Int = 7)
    (ismissing(x) || !isfinite(x)) && return "."
    sig = mx
    s = @sprintf("%.*g", sig, x)
    while length(s) > w && sig > 1
        sig -= 1; s = @sprintf("%.*g", sig, x)
    end
    return 0 < abs(x) < 1 ? replace(s, r"^(-?)0\." => s"\g<1>.") : s
end

# Thousands separators, Stata-style: 1130 -> "1,130".
function _sm_comma(n::Integer)
    s = string(abs(n)); parts = String[]; i = length(s)
    while i >= 1
        push!(parts, s[max(1, i - 2):i]); i -= 3
    end
    return (n < 0 ? "-" : "") * join(reverse(parts), ",")
end

# Stata's `summarize var, detail` block for a single variable.
function _sm_print_detail(name::AbstractString, x::AbstractVector{<:Real},
                          μ::Real, σ, variance, skew, kurt)
    n = length(x)
    ssorted = sort(x)
    small4 = ssorted[1:min(4, n)]
    large4 = ssorted[max(1, n - 3):n]                # ascending, like Stata
    println(); println(lpad(name, 36)); println("-"^61)
    println(lpad("Percentiles", 26), lpad("Smallest", 16))
    pct = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qs  = quantile(x, pct)
    sml = [get(small4, i, missing) for i in 1:4]
    lrg = [get(large4, i, missing) for i in 1:4]
    @printf("%3s%12s%16s\n",              "1%",  _sm_g9(qs[1]), _sm_g9(sml[1]))
    @printf("%3s%12s%16s\n",              "5%",  _sm_g9(qs[2]), _sm_g9(sml[2]))
    @printf("%3s%12s%16s%14s %s\n",       "10%", _sm_g9(qs[3]), _sm_g9(sml[3]),
                                          "Obs", _sm_comma(n))
    @printf("%3s%12s%16s%14s %s\n",       "25%", _sm_g9(qs[4]), _sm_g9(sml[4]),
                                          "Sum of wgt.", _sm_comma(n))
    println()
    @printf("%3s%12s%16s%14s %s\n",       "50%", _sm_g9(qs[5]), "",
                                          "Mean", _sm_g9(μ))
    @printf("%3s%12s%16s%14s %s\n",       "",    "",            "Largest",
                                          "Std. dev.", _sm_g9(σ))
    @printf("%3s%12s%16s\n",              "75%", _sm_g9(qs[6]), _sm_g9(lrg[1]))
    println()
    @printf("%3s%12s%16s%14s %s\n",       "90%", _sm_g9(qs[7]), _sm_g9(lrg[2]),
                                          "Variance", _sm_g9(variance))
    @printf("%3s%12s%16s%14s %s\n",       "95%", _sm_g9(qs[8]), _sm_g9(lrg[3]),
                                          "Skewness", _sm_g9(skew))
    @printf("%3s%12s%16s%14s %s\n",       "99%", _sm_g9(qs[9]), _sm_g9(lrg[4]),
                                          "Kurtosis", _sm_g9(kurt))
end

function stata_summarize(df::AbstractDataFrame, cols = nothing;
                         if_::Union{Nothing,AbstractVector{<:Union{Bool,Missing}}} = nothing,
                         detail::Bool = false)
    # ---- resolve varlist ---------------------------------------------------
    colsyms = if cols === nothing
        [Symbol(n) for n in names(df) if _sm_numeric_vec(df[!, n]) !== nothing]
    elseif cols isa Symbol
        [cols]
    elseif cols isa AbstractString
        _sm_expand_varlist(df, cols)
    elseif cols isa AbstractVector && !isempty(cols) &&
           any(c -> c isa AbstractString, cols)
        _sm_expand_varlist(df, cols)
    else
        collect(cols)                                 # e.g. Vector{Symbol}
    end
    sub = isnothing(if_) ? df : df[coalesce.(if_, false), :]

    names_ = String[]; obs = Int[]; mean_ = Float64[]
    sd = Float64[]; mn = Float64[]; mx = Float64[]
    detail_stats = []                                 # per-variable stats for detail block
    for c in colsyms
        raw = _sm_numeric_vec(sub[!, c])
        v = raw === nothing ? Float64[] : raw
        push!(names_, string(c)); push!(obs, length(v))
        if isempty(v)
            push!(mean_, NaN); push!(sd, NaN); push!(mn, NaN); push!(mx, NaN)
            detail && push!(detail_stats,
                            (name = string(c), x = v, μ = NaN, σ = NaN,
                             variance = NaN, skew = NaN, kurt = NaN))
        else
            μ = mean(v); σ = length(v) > 1 ? std(v) : NaN
            push!(mean_, μ); push!(sd, σ)
            push!(mn, minimum(v)); push!(mx, maximum(v))
            if detail
                # Stata uses biased moments for skewness/kurtosis.
                m2 = mean((v .- μ).^2); m3 = mean((v .- μ).^3); m4 = mean((v .- μ).^4)
                variance = m2 > 0 ? m2 * length(v) / (length(v) - 1) : NaN
                skew = m2 > 0 ? m3 / m2^(3/2) : NaN
                kurt = m2 > 0 ? m4 / m2^2      : NaN
                push!(detail_stats,
                      (name = string(c), x = v, μ = μ, σ = σ,
                       variance = variance, skew = skew, kurt = kurt))
            end
        end
    end

    # ---- detail printing ---------------------------------------------------
    if detail
        for s in detail_stats
            isempty(s.x) ? println("\n$(s.name): no observations") :
                _sm_print_detail(s.name, s.x, s.μ, s.σ, s.variance, s.skew, s.kurt)
        end
        return DataFrame(Variable = names_, Obs = obs, Mean = mean_,
                         Std = sd, Min = mn, Max = mx)
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
