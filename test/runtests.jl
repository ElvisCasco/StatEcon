using StatEcon
using DataFrames, FixedEffectModels, Statistics, Test

# Deterministic little panel: y depends on x, grouped by g, with a 0/1 flag t.
df = DataFrame(
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    x = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0],
    z = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    g = repeat(1:4, inner = 3),
    t = repeat([0, 1], 6),
)

@testset "StatEcon" begin

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


    @testset "bundled datasets" begin
        names = datasets()
        @test !isempty(names)
        @test "auto.dta" in names
        @test isdir(datadir())

        # load by bare stem, by full filename, and check the shape
        auto = dataset("auto")
        @test auto isa DataFrame
        @test size(auto) == (74, 12)
        @test size(dataset("auto.dta")) == size(auto)

        # a stem present as both .dta and .csv resolves to the .dta
        @test endswith(datapath("mus14gdata"), ".dta")
        @test endswith(datapath("mus14gdata.csv"), ".csv")

        # delimited reading, with an explicit header
        f1 = dataset("mus202file1.csv")
        @test f1 isa DataFrame && nrow(f1) > 0
        f2 = dataset("mus202file2.csv"; header = [:name, :age, :female, :income])
        @test propertynames(f2) == [:name, :age, :female, :income]

        # a caret-delimited text file
        psid = dataset("mus02psid92m.txt"; delim = '^',
                       header = [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l])
        @test size(psid, 2) == 12
        @test nrow(psid) > 0

        # unknown names fail loudly, with a hint
        @test_throws ErrorException dataset("no_such_dataset_here")
    end


    @testset "stata_ologit (3+ categories)" begin
        # Regression test for two bugs found by the Wooldridge verification:
        #  1. the cut-point warm start accumulated cumulative shares, so cum > 1
        #     and log(cum/(1-cum)) threw a DomainError for any J >= 3;
        #  2. the inner likelihood accumulators were named `ll`, which Julia bound
        #     to the enclosing `ll` rather than a local, so later calls (the FD
        #     Hessian, then the null fit) clobbered the fitted log-likelihood and
        #     LR chi2 / pseudo R2 always came out 0.
        pension = dataset("pension")
        xs = [:choice, :age, :educ, :female, :black, :married, :finc25,
              :finc35, :finc50, :finc75, :finc100, :finc101, :wealth89, :prftshr]
        r = redirect_stdout(devnull) do
            stata_ologit(pension, :pctstck, xs)      # must not throw
        end
        @test length(r.τ) == 2                        # J = 3 -> 2 cut-points
        @test issorted(r.τ)                           # cut-points increasing
        @test r.ll > r.ll_null                        # fitted beats intercept-only
        @test r.LR > 0                                # not clobbered to 0
        @test r.pseudo_r2 > 0
        @test isapprox(r.LR, 2 * (r.ll - r.ll_null); atol = 1e-8)
        @test isapprox(r.ll, -201.9227; atol = 1e-3)  # verified by hand
    end

end
