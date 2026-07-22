"""
    stata_encode(df, str_col; gen) -> Dict{Int,String}

Stata `encode <str_col>, gen(<gen>)`. Builds a 1-based integer
encoding of the distinct non-missing values in `df[!, str_col]`
(sorted alphabetically — matching Stata's behaviour for string
variables), writes the integer column into `df[!, gen]`, and returns
the label dictionary `Int ⇒ String` (use with `stata_label_list`).
"""
function stata_encode(df::DataFrames.AbstractDataFrame, str_col::Symbol;
                      gen::Symbol)
    col  = df[!, str_col]
    uniq = sort(unique(skipmissing(string.(col))))
    label_of = Dict{String,Int}(s => i for (i, s) in enumerate(uniq))
    df[!, gen] = [ismissing(v) ? missing : label_of[string(v)] for v in col]
    return Dict{Int,String}(i => s for (i, s) in enumerate(uniq))
end
