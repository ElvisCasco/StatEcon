using StatEcon
using DataFrames, FixedEffectModels, Statistics, Test
using GLM

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


    @testset "stata_xtreg_re matches Stata (Wooldridge Ex. 10.4)" begin
        # Two bugs found by the Wooldridge verification:
        #  1. time-invariant regressors (dropped by the within transform) were
        #     still charged a degree of freedom, inflating both variance
        #     components and shifting every RE coefficient;
        #  2. the classical VCE scaled (X*'X*)^-1 by the within-step sigma^2_eps
        #     instead of the theta-demeaned GLS residual variance, leaving every
        #     standard error ~0.2% low.
        # `union` is constant within firm here, which is what triggers (1).
        jt = dataset("jtrain1")
        r = redirect_stdout(devnull) do
            stata_xtreg_re(jt, :lscrap, [:d88, :d89, :union, :grant, :grant_1], :fcode)
        end
        nm = string.(r.coefnames)
        b(v)  = r.β[findfirst(==(v), nm)]
        se(v) = r.se[findfirst(==(v), nm)]

        @test isapprox(b("d88"),     -0.0934519; atol = 5e-7)
        @test isapprox(b("d89"),     -0.2698336; atol = 5e-7)
        @test isapprox(b("union"),    0.5478021; atol = 5e-7)
        @test isapprox(b("grant"),   -0.2146960; atol = 5e-7)
        @test isapprox(b("grant_1"), -0.3770698; atol = 5e-7)
        @test isapprox(b("_cons"),    0.4148333; atol = 5e-7)

        @test isapprox(se("d88"),     0.1091559; atol = 5e-7)
        @test isapprox(se("d89"),     0.1316496; atol = 5e-7)
        @test isapprox(se("union"),   0.4106250; atol = 5e-7)
        @test isapprox(se("grant"),   0.1477838; atol = 5e-7)
        @test isapprox(se("grant_1"), 0.2053516; atol = 5e-7)
        @test isapprox(se("_cons"),   0.2434322; atol = 5e-7)

        # sigma_e equals the FE Root MSE Stata publishes for Example 10.5
        @test isapprox(r.σ_ε, 0.49774421; atol = 5e-8)
        @test isapprox(r.σ_u, 1.39002870; atol = 5e-8)
        @test isapprox(r.σ_u^2 / (r.σ_u^2 + r.σ_ε^2), 0.88634984; atol = 5e-8)
    end


    @testset "stata_regress prints Stata's ANOVA block (Wooldridge Ex. 4.1)" begin
        # Plain `regress` shows Source/SS/df/MS and Adj R-squared; vce(robust)
        # does not. Values are Stata's published output for
        #   reg lwage exper expersq educ age kidslt6 kidsge6   (mroz.dta)
        mroz = dataset("wooldridge/mroz")
        capture(f) = mktemp() do path, io
            redirect_stdout(io) do; f(); end
            flush(io); read(path, String)
        end
        out = capture() do
            stata_regress(mroz; y = :lwage,
                          x = [:exper, :expersq, :educ, :age, :kidslt6, :kidsge6])
        end
        for frag in ("Source |", "Model |", "Residual |", "Total |",
                     "Adj R-squared", "35.3398089", "5.88996815",
                     "187.987632", ".446526442", "223.327441", ".523015084",
                     "0.1582", "0.1462", ".66823")
            @test occursin(frag, out)
        end

        # vce(robust): Stata drops the ANOVA table and Adj R-squared
        rout = capture() do
            stata_regress(mroz; y = :lwage, x = [:exper, :expersq, :educ],
                          vce = :robust)
        end
        @test occursin("Linear regression", rout)
        @test !occursin("Source |", rout)
        @test !occursin("Adj R-squared", rout)
    end


    @testset "stata_probit uses observed information (Wooldridge Ex. 15.2)" begin
        # GLM's vcov is the EXPECTED information from IRLS; Stata's vce(oim)
        # default reports the OBSERVED information. They differ for the
        # non-canonical probit link, which left every probit SE 1-2% high.
        # Values are Stata's published output for
        #   probit inlf nwifeinc educ exper expersq age kidslt6 kidsge6  (mroz)
        mroz = dataset("wooldridge/mroz")
        r = redirect_stdout(devnull) do
            stata_probit(mroz, @formula(inlf ~ nwifeinc + educ + exper +
                                               expersq + age + kidslt6 + kidsge6))
        end
        nm = string.(r.coefnames)
        se(v) = r.se[findfirst(==(v), nm)]
        b(v)  = r.β[findfirst(==(v), nm)]

        # standard errors: the point of the fix, must match to the printed digits
        @test isapprox(se("nwifeinc"), 0.0048398; atol = 5e-7)
        @test isapprox(se("educ"),     0.0252542; atol = 5e-7)
        @test isapprox(se("exper"),    0.0187164; atol = 5e-7)
        @test isapprox(se("age"),      0.0084772; atol = 5e-7)
        @test isapprox(se("kidslt6"),  0.1185223; atol = 5e-7)
        @test isapprox(se("kidsge6"),  0.0434768; atol = 5e-7)
        @test isapprox(se("_cons"),    0.5085930; atol = 5e-7)

        # coefficients agree to ~4e-6 (GLM's IRLS vs Stata's Newton), and the
        # log-likelihood to the printed precision
        @test isapprox(b("nwifeinc"), -0.0120237; atol = 1e-5)
        @test isapprox(b("kidslt6"),  -0.8683285; atol = 1e-5)
        @test isapprox(r.ll, -401.30219; atol = 1e-4)

        # stata_heckman_twostep's stage-1 probit shares the same observed-
        # information vcov, so its selection-block SEs match Stata too (they
        # were ~2% high while it used GLM.vcov directly). Wooldridge Ex. 17.6.
        h = redirect_stdout(devnull) do
            stata_heckman_twostep(mroz,
                @formula(lwage ~ educ + exper + expersq),
                @formula(inlf ~ nwifeinc + educ + exper + expersq + age +
                                kidslt6 + kidsge6);
                depvar_y = "lwage", depvar_d = "inlf")
        end
        snm = string.(h.selection_coefnames)
        @test isapprox(h.se_selection[findfirst(==("nwifeinc"), snm)],
                       0.0048398; atol = 5e-7)
    end


    @testset "robust-VCE finite-sample corrections match Stata" begin
        # stata_glm's robust sandwich used n/(n-k); Stata's glm, vce(robust)
        # uses n/(n-1). Wooldridge Ex. 19.1 (fertil2), educ SE = .0025918.
        fert = dataset("fertil2")
        g = redirect_stdout(devnull) do
            stata_glm(fert, @formula(children ~ educ + age + agesq + evermarr +
                                                urban + electric + tv);
                      family = :poisson, link = :log, vce = :robust)
        end
        gi = findfirst(==("educ"), GLM.coefnames(g.model))
        @test isapprox(g.se[gi], 0.0025918; atol = 5e-7)

        # stata_xtgls iid/independent used n-k; xtgls reports ML variances
        # (divisor n). Wooldridge Ex. 7.7, all four SEs.
        jt77 = dropmissing(dataset("jtrain1"),
                           [:lscrap, :d89, :d88, :grant, :grant_1, :fcode, :year])
        x = redirect_stdout(devnull) do
            stata_xtgls(jt77, :lscrap, [:d89, :d88, :grant, :grant_1];
                        panelvar = :fcode, timevar = :year,
                        panels = :iid, corr = :independent)
        end
        se(v) = x.se[findfirst(==(v), string.(x.coefnames))]
        @test isapprox(se("d89"),     0.3326723; atol = 5e-6)
        @test isapprox(se("d88"),     0.3060290; atol = 5e-6)
        @test isapprox(se("grant"),   0.3330233; atol = 5e-6)
        @test isapprox(se("grant_1"), 0.4292842; atol = 5e-6)
    end


    @testset "stata_oprobit matches Stata (Wooldridge Ex. 15.5)" begin
        # Ordered probit MLE. Values are Stata's published oprobit output for
        #   oprobit pctstck choice age educ female black married finc25 finc35
        #           finc50 finc75 finc100 finc101 wealth89 prftshr   (pension)
        pension = dataset("pension")
        XP = [:choice, :age, :educ, :female, :black, :married, :finc25, :finc35,
              :finc50, :finc75, :finc100, :finc101, :wealth89, :prftshr]
        r = redirect_stdout(devnull) do
            stata_oprobit(pension, :pctstck, XP)
        end
        ci = findfirst(==(:choice), r.regs)
        @test isapprox(r.β[ci],    0.371171;  atol = 1e-4)
        @test isapprox(r.se_β[ci], 0.1841121; atol = 1e-4)
        @test length(r.τ) == 2
        @test isapprox(r.τ[1], -3.087373; atol = 1e-3)
        @test isapprox(r.τ[2], -2.053553; atol = 1e-3)
        @test issorted(r.τ)
        @test isapprox(r.ll,        -201.9865; atol = 1e-2)
        @test isapprox(r.LR,          20.77;   atol = 5e-2)
        @test isapprox(r.pseudo_r2,    0.0489; atol = 1e-3)
    end

    @testset "econ_helpers (lincom/predict_ci/sigma2_of/sargan/…)" begin
        w1 = DataFrame(dataset("wage1"))
        m  = redirect_stdout(devnull) do
            stata_regress(w1; y = :wage, x = [:female, :educ, :exper, :tenure])
        end
        cn = string.(FixedEffectModels.coefnames(m))
        b  = FixedEffectModels.coef(m); V = FixedEffectModels.vcov(m)
        ife = findfirst(==("female"), cn); icn = findfirst(==("(Intercept)"), cn)

        # lincom: unit weight on one coefficient reproduces its own est & se
        r = redirect_stdout(devnull) do; lincom(m, ["female" => 1.0]); end
        @test isapprox(r.est, b[ife];            atol = 1e-10)
        @test isapprox(r.se,  sqrt(V[ife, ife]); atol = 1e-10)
        r2 = redirect_stdout(devnull) do
            lincom(m, ["female" => 1.0, "(Intercept)" => 1.0])
        end
        @test isapprox(r2.est, b[ife] + b[icn];  atol = 1e-10)

        # predict_ci: xb == coef·x0 and se == sqrt(x0' V x0)
        x0 = zeros(length(b)); x0[icn] = 1.0; x0[ife] = 1.0
        pc = redirect_stdout(devnull) do; predict_ci(m, x0); end
        @test isapprox(pc.xb, b[icn] + b[ife];   atol = 1e-10)
        @test isapprox(pc.se, sqrt(x0' * V * x0); atol = 1e-10)

        # sigma2_of == RSS / dof_residual
        @test isapprox(sigma2_of(m, w1, :wage),
                       m.rss / FixedEffectModels.dof_residual(m); rtol = 1e-8)

        # stdbeta returns the fitted model
        mb = redirect_stdout(devnull) do; stdbeta(w1, :wage, [:female, :educ]); end
        @test mb isa FixedEffectModels.FixedEffectModel

        # IV helpers on mroz
        mz = dropmissing(DataFrame(dataset("wooldridge/mroz")),
             [:hours, :lwage, :educ, :age, :kidslt6, :kidsge6, :nwifeinc, :exper, :expersq])
        iv = redirect_stdout(devnull) do
            stata_ivregress_2sls(mz, :hours, :lwage => [:exper, :expersq],
                [:educ, :age, :kidslt6, :kidsge6, :nwifeinc]; robust = false, first = false)
        end
        n = Int(FixedEffectModels.nobs(iv.iv)); k = length(FixedEffectModels.coef(iv.iv))
        se_small = [sqrt(FixedEffectModels.vcov(iv.iv)[i, i]) for i in 1:k]
        tb = redirect_stdout(devnull) do; ivreg2_table(iv.iv; dep = "hours"); end
        # ivreg2_table rescales to large-sample (N divisor) SEs
        @test all(isapprox.(tb.se, se_small .* sqrt((n - k) / n); rtol = 1e-8))

        sg = redirect_stdout(devnull) do
            sargan(mz, :hours, [:lwage],
                   [:educ, :age, :kidslt6, :kidsge6, :nwifeinc], [:exper, :expersq])
        end
        @test sg.df == 1
        @test isfinite(sg.chi2) && 0 <= sg.p <= 1

        # time-series helpers run and return finite values on a sorted series
        d = DataFrame(x = collect(1.0:40.0)); d.y = 2 .+ 0.5 .* d.x .+ 0.4 .* sin.(1:40)
        nw = newey_west(d, :y, [:x], 2)
        @test length(nw.b) == 2 && all(isfinite, nw.se)
        co = redirect_stdout(devnull) do; cochrane_orcutt(d, :y, [:x]); end
        @test isfinite(co.rho) && all(isfinite, co.b)
        @test isfinite(bpagan_lm(d, :y, [:x], [:x]))
    end

    @testset "ordered_classtable reproduces tab pclass (Wooldridge Ex. 15.5)" begin
        XP = [:choice, :age, :educ, :female, :black, :married, :finc25, :finc35,
              :finc50, :finc75, :finc100, :finc101, :wealth89, :prftshr]
        pension = dataset("pension")
        op = redirect_stdout(devnull) do; stata_oprobit(pension, :pctstck, XP); end
        pen = dropmissing(DataFrame(pension)[:, vcat(:pctstck, XP)])
        cats  = sort(unique(Float64.(pen.pctstck)))
        y_idx = [findfirst(==(v), cats) for v in Float64.(pen.pctstck)]
        r = redirect_stdout(devnull) do
            ordered_classtable(op.β, op.τ, op.X, y_idx, cats)
        end
        @test r.ctab == [33 21 11; 25 31 25; 6 20 22]      # Stata `tab pclass pctstck`
        @test sum(r.ctab) == 194
        @test isapprox(r.correct, (33 + 31 + 22) / 194; atol = 1e-8)
        @test all(isapprox.(sum(r.P, dims = 2), 1.0; atol = 1e-10))  # rows are proper distributions
    end

end
