import EnvelopeApproximation.ChebyshevCFT:
    chebyshev_coeffs!, fourier_mode, fourier_modes, scale, translation

using EnvelopeApproximation.ChebyshevCFT
using QuadGK
using Test

@testset "ChebyshevCFT Tests" begin
    @testset "ChebyshevPlan Tests" begin
        # Test fourier transform calculation accuracy with simple functions

        @testset "Constant function" begin
            f(x) = 1.0
            a, b = -1.0, 1.0

            # Analytical Fourier transform of constant function: ∫e^(-ikx)dx = 2*sin(k)/(k)
            expected_ft(k) = k ≈ 0.0 ? 2.0 + 0.0im : 2 * sin(k) / k

            plan = ChebyshevPlan{8192}()
            ks = [0.0, 0.5, 1.0, 2.0, 5.0]

            for k in ks
                # Compute with ChebyshevCFT
                chebyshev_coeffs!(f, a, b, plan)
                result = fourier_mode(k, plan, scale(a, b), translation(a, b))
                # Compute exact
                exact = expected_ft(k)
                @test isapprox(result, exact, rtol=1e-8)
            end
        end

        @testset "Gaussian function" begin
            σ = 0.5
            μ = 0.0
            f(x) = exp(-(x - μ)^2 / (2 * σ^2))
            a, b = -3.0, 3.0

            # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
            expected_ft(k) = σ * sqrt(2 * π) * exp(-k^2 * σ^2 / 2) * exp(-im * k * μ)

            plan = ChebyshevPlan{64}()
            ks = [0.0, 0.5, 1.0, 2.0, 5.0]

            for k in ks
                chebyshev_coeffs!(f, a, b, plan)
                # Our computed transform
                result = fourier_mode(k, plan, scale(a, b), translation(a, b))

                # The exact transform would be over the infinite domain
                # But our function is effectively zero outside our interval
                numerical, _ = quadgk(x -> f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                @test isapprox(result, numerical, rtol=1e-8)

                # For small k, we can compare with the analytical result
                if abs(k) < 3.0
                    @test isapprox(result, expected_ft(k), rtol=1e-3)
                end
            end
        end
    end


    @testset "First3MomentsChebyshevPlan Accuracy" begin
        f(x) = exp(x)
        a, b = 2.0, 3.0

        plan = First3MomentsChebyshevPlan{8192}()
        chebyshev_coeffs!(f, a, b, plan)

        ks = [0.0, 0.5, 1.0, 2.0]

        for k in ks
            # Get the 0th, 1st, and 2nd moments
            m0, m1, m2 = fourier_mode(k, plan, scale(a, b), translation(a, b))

            # Direct numerical integration for 0th moment
            ft0, _ = quadgk(x -> f(x) * exp(-im * k * x), a, b, rtol=1e-12)
            @test isapprox(m0, ft0, rtol=1e-6)

            # Direct numerical integration for 1st moment
            ft1, _ = quadgk(x -> x * f(x) * exp(-im * k * x), a, b, rtol=1e-12)
            @test isapprox(m1, ft1, rtol=1e-6)

            # Direct numerical integration for 2nd moment
            ft2, _ = quadgk(x -> x^2 * f(x) * exp(-im * k * x), a, b, rtol=1e-12)
            @test isapprox(m2, ft2, rtol=1e-6)
        end
    end

    @testset "VectorChebyshevPlan Accuracy" begin
        f(x) = [exp(x), sin(x), cos(x)]
        a, b = 1.0, 3.0

        plan = VectorChebyshevPlan{8192,3}()
        chebyshev_coeffs!(f, a, b, plan)

        ks = [0.0, 0.5, 1.0, 2.0]

        for k in ks
            result = fourier_mode(k, plan, scale(a, b), translation(a, b))

            # Check each component separately
            for i in 1:3
                component_f(x) = f(x)[i]
                numerical, _ = quadgk(x -> component_f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                @test isapprox(result[i], numerical, rtol=1e-7)
            end
        end
    end

    @testset "TailoredChebyshevPlan Tests" begin
        @testset "Constant function" begin
            f(x) = 1.0
            a, b = -1.0, 1.0

            # Analytical Fourier transform of constant function: ∫e^(-ikx)dx = 2*sin(k)/(k)
            expected_ft(k) = k ≈ 0.0 ? 2.0 + 0.0im : 2 * sin(k) / k

            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredChebyshevPlan{8192}(ks, a, b)

            # Compute with TailoredChebyshevPlan
            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)

            # Test each value
            for (i, k) in enumerate(ks)
                exact = expected_ft(k)
                @test isapprox(results[i], exact, rtol=1e-8)
            end
        end

        @testset "Gaussian function" begin
            σ = 0.5
            μ = 0.0
            f(x) = exp(-(x - μ)^2 / (2 * σ^2))
            a, b = -3.0, 3.0

            # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
            expected_ft(k) = σ * sqrt(2 * π) * exp(-k^2 * σ^2 / 2) * exp(-im * k * μ)

            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredChebyshevPlan{64}(ks, a, b)

            # Compute with TailoredChebyshevPlan
            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)

            for (i, k) in enumerate(ks)
                # The exact transform would be over the infinite domain
                # But our function is effectively zero outside our interval
                numerical, _ = quadgk(x -> f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                @test isapprox(results[i], numerical, rtol=1e-8)

                # For small k, we can compare with the analytical result
                if abs(k) < 3.0
                    @test isapprox(results[i], expected_ft(k), rtol=1e-3)
                end
            end
        end

        @testset "Multiple k values simultaneously" begin
            # Test that we can compute multiple k values at once efficiently
            f(x) = sin(x) + cos(2x)
            a, b = 0.0, Float64(π)

            # Test with a large number of k values
            ks = collect(0.0:0.25:10.0)
            plan = TailoredChebyshevPlan{4096}(ks, a, b)

            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)

            # Check a random subset for accuracy
            for idx in [1, 5, 10, 20, 40]
                k = ks[idx]
                numerical, _ = quadgk(x -> f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                @test isapprox(results[idx], numerical, rtol=1e-6)
            end
        end
    end

    @testset "TailoredVectorChebyshevPlan Tests" begin

        @testset "Constant vector function" begin
            # Vector-valued constant function
            f(x) = [1.0, 2.0, 3.0]
            a, b = -1.0, 1.0

            # Analytical Fourier transform of constant function: ∫e^(-ikx)dx = 2*sin(k)/(k)
            expected_ft(k, val) = k ≈ 0.0 ? 2.0 * val + 0.0im : 2 * sin(k) / k * val

            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredVectorChebyshevPlan{8192,3}(ks, a, b)

            # Compute with TailoredVectorChebyshevPlan
            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)

            # Test each value and component
            for (i, k) in enumerate(ks)
                for j in 1:3
                    exact = expected_ft(k, f(0.0)[j])
                    @test isapprox(results[i, j], exact, rtol=1e-8)
                end
            end
        end

        @testset "Vector of Gaussian functions" begin
            # Define a vector of Gaussian functions with different parameters
            σ_values = [0.5, 0.8, 1.0]
            μ_values = [0.0, 0.5, -0.5]

            f(x) = [exp(-(x - μ)^2 / (2 * σ^2)) for (σ, μ) in zip(σ_values, μ_values)]

            a, b = -5.0, 5.0  # Large enough domain for all gaussians

            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredVectorChebyshevPlan{4096,3}(ks, a, b)

            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)

            # Test each gaussian component
            for (i, k) in enumerate(ks)
                for j in 1:3
                    σ, μ = σ_values[j], μ_values[j]
                    # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
                    expected = σ * sqrt(2 * π) * exp(-k^2 * σ^2 / 2) * exp(-im * k * μ)

                    gaussian_f(x) = exp(-(x - μ)^2 / (2 * σ^2))
                    numerical, _ = quadgk(x -> gaussian_f(x) * exp(-im * k * x), a, b, rtol=1e-12)

                    @test isapprox(results[i, j], numerical, rtol=1e-6)

                    # For small k, we can compare with the analytical result
                    if abs(k) < 3.0
                        @test isapprox(results[i, j], expected, rtol=1e-3)
                    end
                end
            end
        end
    end

    @testset "ChebyshevPlanWithAtol Tests" begin
        @testset "Basic functionality" begin
            f(x) = cos(x)
            a, b = -1.0, 1.0

            # Create a plan with α=2 (typical for smooth functions), P must be odd
            plan = ChebyshevPlanWithAtol{3 ^ 8,3}(2.0)

            ks = [0.0, 0.5, 1.0, 2.0, 5.0]

            # Compute Fourier transform
            modes, error_estimate = fourier_modes(f, ks, a, b, plan)

            # Verify against direct numerical integration
            for (i, k) in enumerate(ks)
                numerical, _ = quadgk(x -> f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                @test isapprox(modes[i], numerical, rtol=1e-6)
            end
        end

        @testset "Error estimation" begin
            # Test how error estimate changes with different parameters

            # Highly oscillatory function (should require more points)
            f(x) = sin(10 * x)
            a, b = -π, π

            # Test with different N values, P must be odd, N must be divisible by P
            plan1 = ChebyshevPlanWithAtol{33,3}(2.0)  # 33 ÷ 3 = 11
            plan2 = ChebyshevPlanWithAtol{63,3}(2.0)  # 63 ÷ 3 = 21
            plan3 = ChebyshevPlanWithAtol{123,3}(2.0) # 123 ÷ 3 = 41

            ks = [1.0, 2.0]

            _, error1 = fourier_modes(f, ks, a, b, plan1)
            _, error2 = fourier_modes(f, ks, a, b, plan2)
            _, error3 = fourier_modes(f, ks, a, b, plan3)

            # Error should decrease with increasing N
            @test error1 > error2
            @test error2 > error3
        end

        @testset "Convergence rate parameter α" begin
            # Smooth function with different α settings
            f(x) = exp(x)
            a, b = -1.0, 1.0
            ks = [0.5, 1.0]

            # Test with different α values, P must be odd, N must be divisible by P
            plan1 = ChebyshevPlanWithAtol{63,3}(1.0)  # α=1, slower convergence
            plan2 = ChebyshevPlanWithAtol{63,3}(2.0)  # α=2, faster convergence
            plan3 = ChebyshevPlanWithAtol{63,3}(3.0)  # α=3, even faster convergence

            # Compute with each plan
            modes1, error1 = fourier_modes(f, ks, a, b, plan1)
            modes2, error2 = fourier_modes(f, ks, a, b, plan2)
            modes3, error3 = fourier_modes(f, ks, a, b, plan3)

            # Error estimates should decrease with increasing α (due to formula error_estimate = inf_norm / (P^α - 1))
            @test error1 > error2
            @test error2 > error3

            # Results should be similar regardless of α (since α only affects error estimation)
            for i in 1:length(ks)
                @test isapprox(modes1[i], modes2[i], rtol=1e-8)
                @test isapprox(modes2[i], modes3[i], rtol=1e-8)
            end
        end

        @testset "Constructor constraints" begin
            # Test that constructor enforces P must be odd and N must be divisible by P
            f(x) = sin(x)
            a, b = -1.0, 1.0

            # P must be odd
            @test_throws ArgumentError ChebyshevPlanWithAtol{63,2}(2.0)

            # N must be divisible by P
            @test_throws ArgumentError ChebyshevPlanWithAtol{64,3}(2.0)

            # These should work fine
            N = 2^10
            plan3 = ChebyshevPlanWithAtol{N * 3,3}(2.0) 
            plan5 = ChebyshevPlanWithAtol{N * 5,5}(2.0) 
            plan7 = ChebyshevPlanWithAtol{N * 7,7}(2.0)  #

            # Check that they all give consistent results
            ks = [1.0]
            modes3, _ = fourier_modes(f, ks, a, b, plan3)
            modes5, _ = fourier_modes(f, ks, a, b, plan5)
            modes7, _ = fourier_modes(f, ks, a, b, plan7)

            @test isapprox(modes3[1], modes5[1], atol=1e-7)
            @test isapprox(modes5[1], modes7[1], atol=1e-7)
        end

        @testset "Warning generation" begin
            # Test that warnings are generated when error exceeds tolerance
            f(x) = sin(20 * x) # Highly oscillatory function to trigger warnings with small N
            a, b = -π, π

            # Set a very small tolerance that should be exceeded
            # N=33 is divisible by P=3
            plan = ChebyshevPlanWithAtol{33,3}(2.0, atol=1e-12)
            ks = [1.0]

            # Should generate a warning
            @test_logs (:warn, r"Chebyshev approximation error .* exceeds tolerance .*") begin
                _, _ = fourier_modes(f, ks, a, b, plan)
            end
        end

        @testset "Comparison with standard ChebyshevPlan" begin
            # Test that results match with standard ChebyshevPlan for same N
            f(x) = exp(-x^2)
            a, b = -3.0, 3.0
            ks = [0.0, 1.0, 2.0]

            # N=63 is divisible by P=3
            std_plan = ChebyshevPlan{63}()
            atol_plan = ChebyshevPlanWithAtol{63,3}(2.0)

            # Compute with standard plan
            chebyshev_coeffs!(f, a, b, std_plan)
            s = scale(a, b)
            t = translation(a, b)
            std_results = [fourier_mode(k, std_plan, s, t) for k in ks]

            # Compute with atol plan
            atol_results, _ = fourier_modes(f, ks, a, b, atol_plan)

            # Results should match
            for i in 1:length(ks)
                @test isapprox(std_results[i], atol_results[i], rtol=1e-8)
            end
        end
    end

    @testset "VectorChebyshevPlanWithAtol Tests" begin
        @testset "Basic functionality" begin
            # Vector-valued function test
            f(x) = [sin(x), cos(x), exp(-x^2)]
            a, b = -2.0, 2.0

            # P must be odd, N must be divisible by P (63 ÷ 3 = 21)
            plan = VectorChebyshevPlanWithAtol{3 ^ 8,3,3}(2.0)
            ks = [0.0, 0.5, 1.0, 2.0]

            # Compute Fourier transform
            modes, error_estimate = fourier_modes(f, ks, a, b, plan)

            # Verify against direct numerical integration for each component
            for (i, k) in enumerate(ks)
                for j in 1:3
                    component_f(x) = f(x)[j]
                    numerical, _ = quadgk(x -> component_f(x) * exp(-im * k * x), a, b, rtol=1e-12)
                    @test isapprox(modes[i, j], numerical, atol=1e-15, rtol=1e-6)
                end
            end
        end

        @testset "Error estimation" begin
            # Vector-valued function with different N values
            f(x) = [sin(5 * x), cos(5 * x)]
            a, b = -π, π

            # Test with different N values, P must be odd, N must be divisible by P
            plan1 = VectorChebyshevPlanWithAtol{33,2,3}(2.0)  # 33 ÷ 3 = 11
            plan2 = VectorChebyshevPlanWithAtol{63,2,3}(2.0)  # 63 ÷ 3 = 21
            plan3 = VectorChebyshevPlanWithAtol{99,2,3}(2.0)  # 99 ÷ 3 = 33

            ks = [1.0, 2.0]

            _, error1 = fourier_modes(f, ks, a, b, plan1)
            _, error2 = fourier_modes(f, ks, a, b, plan2)
            _, error3 = fourier_modes(f, ks, a, b, plan3)

            # Error should decrease with increasing N
            @test error1 > error2
            @test error2 > error3
        end

        @testset "Constructor constraints" begin
            # Test constructor enforces constraints: P must be odd and N must be divisible by P
            f(x) = [sin(x), cos(x)]
            a, b = -1.0, 1.0

            # P must be odd
            @test_throws ArgumentError VectorChebyshevPlanWithAtol{63,2,2}(2.0)

            # N must be divisible by P
            @test_throws ArgumentError VectorChebyshevPlanWithAtol{64,2,3}(2.0)

            # These should work fine
            plan3 = VectorChebyshevPlanWithAtol{3 ^ 8,2,3}(2.0)  # 63 ÷ 3 = 21
            plan5 = VectorChebyshevPlanWithAtol{5 ^ 6,2,5}(2.0)  # 65 ÷ 5 = 13

            # Check that they all give consistent results
            ks = [1.0]
            modes3, _ = fourier_modes(f, ks, a, b, plan3)
            modes5, _ = fourier_modes(f, ks, a, b, plan5)

            for j in 1:2
                @test isapprox(modes3[1, j], modes5[1, j], rtol=1e-7)
            end
        end

        @testset "Comparison with standard VectorChebyshevPlan" begin
            f(x) = [exp(-x^2), sin(x)]
            a, b = -3.0, 3.0
            ks = [0.0, 1.0]

            # N=63 is divisible by P=3
            std_plan = VectorChebyshevPlan{63,2}()
            atol_plan = VectorChebyshevPlanWithAtol{63,2,3}(2.0)

            # Compute with standard plan
            std_results = fourier_modes(f, ks, a, b, std_plan)

            # Compute with atol plan
            atol_results, _ = fourier_modes(f, ks, a, b, atol_plan)

            # Results should match
            for i in 1:length(ks)
                for j in 1:2
                    @test isapprox(std_results[i, j], atol_results[i, j], rtol=1e-8)
                end
            end
        end

        @testset "Parameter P impact" begin
            # Test how P parameter affects error estimation
            f(x) = [sin(x), cos(x)]
            a, b = -π, π
            ks = [1.0]

            # Different P values (N must be divisible by P), P must be odd
            plan1 = VectorChebyshevPlanWithAtol{63,2,3}(2.0)  # 63 ÷ 3 = 21
            plan2 = VectorChebyshevPlanWithAtol{63,2,9}(2.0)  # 63 ÷ 9 = 7

            _, error1 = fourier_modes(f, ks, a, b, plan1)
            _, error2 = fourier_modes(f, ks, a, b, plan2)

            # For same α, larger P should give larger error estimate (since P^α - 1 is smaller)
            @test error1 < error2
        end

        @testset "Warning generation" begin
            # Test that warnings are generated when error exceeds tolerance
            f(x) = [sin(20 * x), cos(20 * x)] # Highly oscillatory
            a, b = -π, π

            # P must be odd, N must be divisible by P
            plan = VectorChebyshevPlanWithAtol{33,2,3}(2.0, atol=1e-12)  # 33 ÷ 3 = 11
            ks = [1.0]

            @test_logs (:warn, r"Vector Chebyshev approximation error .* exceeds tolerance .*") begin
                _, _ = fourier_modes(f, ks, a, b, plan)
            end
        end
    end
end