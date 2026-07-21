# (deps provided by the Stat_Econ module)

# Gaussian kernel-density estimate, dependency-free (Statistics only).
# Mimics Stata's `kdensity` closely enough for overlap visualisation and is
# reused by the `teffects overlap` translation.
#
#   x, d = kdensity(v)                 # evaluate on an automatic grid
#   x, d = kdensity(v; npoints = 100)  # control grid resolution
#
# Bandwidth defaults to Silverman's rule of thumb
#   h = 0.9 * min(std, IQR/1.349) * n^(-1/5),
# the same optimal-for-Gaussian rule Stata's kdensity uses by default.
function kdensity(v::AbstractVector; npoints::Int = 200,
                  bandwidth::Union{Nothing,Real} = nothing, cut::Real = 3.0)
    x = Float64.(collect(skipmissing(v)))
    n = length(x)
    s   = std(x)
    iqr = quantile(x, 0.75) - quantile(x, 0.25)
    h = isnothing(bandwidth) ? 0.9 * min(s, iqr / 1.349) * n^(-1/5) : float(bandwidth)
    h = h > 0 ? h : (s > 0 ? s : 1.0)                       # guard degenerate spread

    grid = range(minimum(x) - cut * h, maximum(x) + cut * h, length = npoints)
    dens = [sum(exp.(-0.5 .* ((g .- x) ./ h) .^ 2)) / (n * h * sqrt(2π)) for g in grid]
    return collect(grid), dens
end
