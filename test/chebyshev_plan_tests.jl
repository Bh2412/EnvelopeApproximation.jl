using Test
using EnvelopeApproximation.ChebyshevCFT
import EnvelopeApproximation.ChebyshevCFT: chebyshev_coeffs!, fourier_mode, scale, translation, fourier_modes
using QuadGK

@testset "ChebyshevCFT Tests" begin
    # Test fourier transform calculation accuracy with simple functions
    
    @testset "Constant function" begin
        f(x) = 1.0
        a, b = -1.0, 1.0
        
        # Analytical Fourier transform of constant function: ∫e^(-ikx)dx = 2*sin(k)/(k)
        expected_ft(k) = k ≈ 0.0 ? 2.0 + 0.0im : 2*sin(k)/k
        
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
        f(x) = exp(-(x-μ)^2/(2*σ^2))
        a, b = -3.0, 3.0
        
        # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
        expected_ft(k) = σ*sqrt(2*π)*exp(-k^2*σ^2/2)*exp(-im*k*μ)
        
        plan = ChebyshevPlan{64}()
        ks = [0.0, 0.5, 1.0, 2.0, 5.0]
        
        for k in ks
            chebyshev_coeffs!(f, a, b, plan)
            # Our computed transform
            result = fourier_mode(k, plan, scale(a, b), translation(a, b))
            
            # The exact transform would be over the infinite domain
            # But our function is effectively zero outside our interval
            numerical, _ = quadgk(x -> f(x) * exp(-im*k*x), a, b, rtol=1e-12)
            @test isapprox(result, numerical, rtol=1e-8)
            
            # For small k, we can compare with the analytical result
            if abs(k) < 3.0
                @test isapprox(result, expected_ft(k), rtol=1e-3)
            end
        end
    end
    
    
    @testset "First3MomentsChebyshevPlan Accuracy" begin
        f(x) = exp(x)
        a, b = 2., 3.
        
        plan = First3MomentsChebyshevPlan{8192}()
        chebyshev_coeffs!(f, a, b, plan)
        
        ks = [0.0, 0.5, 1.0, 2.0]
        
        for k in ks
            # Get the 0th, 1st, and 2nd moments
            m0, m1, m2 = fourier_mode(k, plan, scale(a, b), translation(a, b))
            
            # Direct numerical integration for 0th moment
            ft0, _ = quadgk(x -> f(x) * exp(-im*k*x), a, b, rtol=1e-12)
            @test isapprox(m0, ft0, rtol=1e-6)
            
            # Direct numerical integration for 1st moment
            ft1, _ = quadgk(x -> x * f(x) * exp(-im*k*x), a, b, rtol=1e-12)
            @test isapprox(m1, ft1, rtol=1e-6)
            
            # Direct numerical integration for 2nd moment
            ft2, _ = quadgk(x -> x^2 * f(x) * exp(-im*k*x), a, b, rtol=1e-12)
            @test isapprox(m2, ft2, rtol=1e-6)
        end
    end
    
    @testset "VectorChebyshevPlan Accuracy" begin
        f(x) = [exp(x), sin(x), cos(x)]
        a, b = 1., 3.
        
        plan = VectorChebyshevPlan{8192, 3}()
        chebyshev_coeffs!(f, a, b, plan)
        
        ks = [0.0, 0.5, 1.0, 2.0]
        
        for k in ks
            result = fourier_mode(k, plan, scale(a, b), translation(a, b))
            
            # Check each component separately
            for i in 1:3
                component_f(x) = f(x)[i]
                numerical, _ = quadgk(x -> component_f(x) * exp(-im*k*x), a, b, rtol=1e-12)
                @test isapprox(result[i], numerical, rtol=1e-7)
            end
        end
    end

    @testset "TailoredChebyshevPlan Tests" begin
        @testset "Constant function" begin
            f(x) = 1.0
            a, b = -1.0, 1.0
            
            # Analytical Fourier transform of constant function: ∫e^(-ikx)dx = 2*sin(k)/(k)
            expected_ft(k) = k ≈ 0.0 ? 2.0 + 0.0im : 2*sin(k)/k
            
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
            f(x) = exp(-(x-μ)^2/(2*σ^2))
            a, b = -3.0, 3.0
            
            # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
            expected_ft(k) = σ*sqrt(2*π)*exp(-k^2*σ^2/2)*exp(-im*k*μ)
            
            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredChebyshevPlan{64}(ks, a, b)
            
            # Compute with TailoredChebyshevPlan
            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)
            
            for (i, k) in enumerate(ks)
                # The exact transform would be over the infinite domain
                # But our function is effectively zero outside our interval
                numerical, _ = quadgk(x -> f(x) * exp(-im*k*x), a, b, rtol=1e-12)
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
                numerical, _ = quadgk(x -> f(x) * exp(-im*k*x), a, b, rtol=1e-12)
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
            expected_ft(k, val) = k ≈ 0.0 ? 2.0*val + 0.0im : 2*sin(k)/k * val
            
            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredVectorChebyshevPlan{8192, 3}(ks, a, b)
            
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
            
            f(x) = [exp(-(x-μ)^2/(2*σ^2)) for (σ, μ) in zip(σ_values, μ_values)]
            
            a, b = -5.0, 5.0  # Large enough domain for all gaussians
            
            ks = [0.0, 0.5, 1.0, 2.0, 5.0]
            plan = TailoredVectorChebyshevPlan{4096, 3}(ks, a, b)
            
            chebyshev_coeffs!(f, plan)
            results = fourier_modes(plan)
            
            # Test each gaussian component
            for (i, k) in enumerate(ks)
                for j in 1:3
                    σ, μ = σ_values[j], μ_values[j]
                    # Analytical Fourier transform of Gaussian: σ*sqrt(2π)*exp(-k^2*σ^2/2)*exp(-ikμ)
                    expected = σ*sqrt(2*π)*exp(-k^2*σ^2/2)*exp(-im*k*μ)
                    
                    gaussian_f(x) = exp(-(x-μ)^2/(2*σ^2))
                    numerical, _ = quadgk(x -> gaussian_f(x) * exp(-im*k*x), a, b, rtol=1e-12)
                    
                    @test isapprox(results[i, j], numerical, rtol=1e-6)
                    
                    # For small k, we can compare with the analytical result
                    if abs(k) < 3.0
                        @test isapprox(results[i, j], expected, rtol=1e-3)
                    end
                end
            end
        end
    end
    
end