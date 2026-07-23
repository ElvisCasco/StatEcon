# =============================================================================
# Bundled datasets — load the data shipped with StatEcon by name, with no
# reference to the on-disk folder layout:
#
#     auto  = dataset("auto")            # -> DataFrame  (data/auto.dta)
#     df10  = dataset("mus10data")       # -> DataFrame  (data/musr/mus10data.dta)
#     datasets()                         # -> what is available
#     datapath("mus202file2.csv")        # -> absolute path (exotic parsing / writing)
#
# `.dta` is read with ReadStatTables; `.csv` / `.txt` with the DelimitedFiles
# stdlib, so no extra dependency is required.
# =============================================================================

"""
    datadir() -> String

Absolute path of the data directory bundled with StatEcon.
"""
datadir() = normpath(joinpath(@__DIR__, "..", "data"))

# Every bundled file as (relative-path-under-data, absolute-path). Relative
# paths use '/' on all platforms so subfolder-qualified lookups are portable.
function _all_data_files()
    root = datadir()
    out = Tuple{String,String}[]
    isdir(root) || return out
    for (dir, _, files) in walkdir(root), f in files
        full = joinpath(dir, f)
        rel  = replace(relpath(full, root), '\\' => '/')
        push!(out, (rel, full))
    end
    return out
end

"""
    datasets() -> Vector{String}

Sorted list of every bundled data file, as its path relative to the data
directory (e.g. `"auto.dta"`, `"musr/mus10data.dta"`,
`"wooldridge_intro/wage1.dta"`). Pass any of these to [`dataset`](@ref) /
[`datapath`](@ref) with or without the extension, and — when a bare name is
shared by more than one folder — with the folder prefix to disambiguate.
"""
datasets() = sort(first.(_all_data_files()))

"""
    datapath(name) -> String

Absolute path of a bundled data file, looked up by name with or without its
extension: `datapath("auto")`, `datapath("mus202file2.csv")`, or a
folder-qualified `datapath("wooldridge_intro/mroz")`.

A bare name that resolves to files in more than one folder (e.g. `"mroz"`,
which the panel and introductory Wooldridge sets both ship, with different
contents) raises an error listing the candidates rather than silently
returning one — qualify it with the folder. A stem shared only across
extensions in the same folder (e.g. `mus14gdata.dta`/`.csv`) resolves to the
`.dta` unless the extension is given.
"""
function datapath(name::AbstractString)
    q = replace(String(name), '\\' => '/')
    files = _all_data_files()
    _stem(p) = first(splitext(p))
    matches(rel) = q == rel || q == _stem(rel) ||
                   q == basename(rel) || q == _stem(basename(rel))
    cands = unique([full for (rel, full) in files if matches(rel)])

    if length(cands) > 1
        # prefer .dta when the collision is only across extensions
        dtas = filter(p -> lowercase(splitext(p)[2]) == ".dta", cands)
        length(dtas) == 1 && (cands = dtas)
    end

    if length(cands) == 1
        return cands[1]
    elseif length(cands) > 1
        rels = sort([replace(relpath(p, datadir()), '\\' => '/') for p in cands])
        error("Ambiguous dataset \"$name\" — it matches several bundled files:\n  " *
              join(rels, "\n  ") *
              "\nQualify it with the folder, e.g. dataset(\"$(rels[1])\").")
    else
        qstem = lowercase(_stem(basename(q)))
        near = sort(unique([_stem(basename(rel)) for (rel, _) in files
                            if occursin(qstem, lowercase(_stem(basename(rel))))]))
        hint = isempty(near) ? "Call `datasets()` to list what is bundled." :
                               "Did you mean: " * join(first(near, 5), ", ") * "?"
        error("No bundled dataset named \"$name\". $hint")
    end
end

# Coerce a readdlm column to a concrete numeric vector when every entry parses.
function _ds_coerce(col::AbstractVector)
    vals = Any[x isa AbstractString ? strip(x) : x for x in col]
    if all(v -> v isa Real, vals)
        xs = Float64[Float64(v) for v in vals]
        all(x -> isinteger(x) && abs(x) < 9.0e15, xs) && return [Int(x) for x in xs]
        return xs
    end
    nums = Union{Float64,Nothing}[v isa Real ? Float64(v) :
                                  (v isa AbstractString ? tryparse(Float64, v) : nothing)
                                  for v in vals]
    any(isnothing, nums) && return [v isa AbstractString ? String(v) : v for v in vals]
    xs = Float64[n for n in nums]
    all(x -> isinteger(x) && abs(x) < 9.0e15, xs) && return [Int(x) for x in xs]
    return xs
end

function _ds_read_delimited(path::AbstractString;
                            delim = nothing,
                            header::Union{Bool,AbstractVector} = true)
    d = delim === nothing ?
        (lowercase(splitext(path)[2]) == ".csv" ? ',' : _ds_guess_delim(path)) : delim
    if header === true
        raw, hdr = DelimitedFiles.readdlm(path, d, Any; header = true)
        names_ = [Symbol(strip(string(h))) for h in vec(hdr)]
    else
        raw = DelimitedFiles.readdlm(path, d, Any)
        names_ = header === false ?
            [Symbol("Column", j) for j in 1:size(raw, 2)] :
            [Symbol(h) for h in header]
    end
    size(raw, 2) == length(names_) ||
        error("datasets: $(basename(path)) has $(size(raw,2)) columns but " *
              "$(length(names_)) names were supplied.")
    return DataFrames.DataFrame([names_[j] => _ds_coerce(raw[:, j])
                                 for j in 1:size(raw, 2)])
end

function _ds_guess_delim(path::AbstractString)
    line = ""
    open(path) do io
        for l in eachline(io)
            isempty(strip(l)) && continue
            line = l; break
        end
    end
    for c in ('^', '\t', ';', '|', ',')
        occursin(c, line) && return c
    end
    return ' '
end

"""
    dataset(name; kwargs...) -> DataFrame

Load a dataset bundled with StatEcon by name, without referring to the data
folder:

```julia
auto = dataset("auto")           # data/auto.dta
df10 = dataset("mus10data")      # data/musr/mus10data.dta
```

The name may include the extension (`dataset("mus14gdata.csv")`); when a stem
exists in several formats the `.dta` version is chosen. `.dta` files are read
with ReadStatTables, `.csv`/`.txt` with DelimitedFiles.

For delimited files, `delim` and `header` are forwarded: `header=true` (default)
takes the first row as column names, `header=false` generates `Column1…`, and a
vector supplies them explicitly:

```julia
psid = dataset("mus02psid92m.txt"; delim = '^',
               header = [:famid, :year, :educ])
```

See also [`datasets`](@ref), [`datapath`](@ref), [`datadir`](@ref).
"""
function dataset(name::AbstractString; kwargs...)
    path = datapath(name)
    ext  = lowercase(splitext(path)[2])
    if ext == ".dta"
        isempty(kwargs) ||
            @warn "dataset(\"$name\"): keyword arguments are ignored for .dta files."
        return DataFrames.DataFrame(ReadStatTables.readstat(path))
    elseif ext in (".csv", ".txt", ".tsv", ".asc", ".raw")
        return _ds_read_delimited(path; kwargs...)
    else
        error("dataset: don't know how to read \"$(basename(path))\" (extension $ext).")
    end
end
