using Stat_Econ
using DataFrames, FixedEffectModels, Statistics, Test

# Deterministic little panel: y depends on x, grouped by g, with a 0/1 flag t.
df = DataFrame(
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    x = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0],
    z = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    g = repeat(1:4, inner = 3),
    t = repeat([0, 1], 6),
)

@testset "Stat_Econ" begin

    @testset "stata_regress returns a usable model" begin
        m = redirect_stdout(devnull) do
            stata_regress(df, y = :y, x = [:x], vce = :robust)
        end
        @test m isa FixedEffectModels.FixedEffectModel
        @test length(FixedEffectModels.coef(m)) == 2          # x + intercept
        @test FixedEffectModels.nobs(m) == nrow(df)
    end

    @testset "vce options" begin
        for (vce, kw) in ((:ols, ()), (:robust, ()), (:cluster, (cluster = :g,)))
            m = redirect_stdout(devnull) do
                stata_regress(df; y = :y, x = [:x], vce = vce, kw...)
            end
            @test FixedEffectModels.nobs(m) == nrow(df)
        end
    end

    @testset "nocons drops the intercept" begin
        m = redirect_stdout(devnull) do
            stata_regress(df, y = :y, x = [:x], nocons = true, vce = :ols)
        end
        @test length(FixedEffectModels.coef(m)) == 1
    end

    @testset "absorb gives a within estimator" begin
        m = redirect_stdout(devnull) do
            stata_regress(df, y = :y, x = [:x], absorb = :g, vce = :cluster, cluster = :g)
        end
        @test m.dof_fes > 0
    end

    @testset "interactions" begin
        m = redirect_stdout(devnull) do
            stata_regress(df, y = :y, x = [:x, (:x, :z)], vce = :ols)
        end
        @test any(n -> occursin("&", n), FixedEffectModels.coefnames(m))
    end

    @testset "stata_summarize" begin
        out = redirect_stdout(devnull) do
            stata_summarize(df, :x)
        end
        @test out.Obs[1] == nrow(df)
        @test out.Mean[1] ≈ mean(df.x)
    end

    @testset "stata_tabulate" begin
        out = redirect_stdout(devnull) do
            stata_tabulate(df, :t)
        end
        @test sum(out.Freq) == nrow(df)
        @test out.Cum[end] ≈ 100.0
    end

    @testset "kdensity integrates to ~1" begin
        xs, ds = kdensity(df.x)
        @test length(xs) == length(ds)
        @test all(>=(0), ds)
        area = sum(diff(xs) .* ds[1:end-1])
        @test 0.85 < area < 1.15
    end

end
