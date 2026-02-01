using Test
using EnvelopeApproximation.SphericalHarmonics

@testset "Spherical Harmonic Transform Tests" begin

    @testset "Constant function" begin
        # f = 1 should give only Y₀₀
        f_const(ϕ, θ) = [1.0]
        coeffs = spherical_harmonic_coefficients(f_const, 20, 1)
        
        # Y₀₀ should be non-zero
        @test abs(coeffs[1, 1, 1]) ≈ sqrt(4π)
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs[:, :, 1])
        coeffs_copy[1, 1] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

    @testset "cos(θ) ~ Y₁₀" begin
        # cos(θ) = √(4π/3) Y₁₀
        # Y₁₀ means ℓ=1, m=0
        f_costheta(ϕ, θ) = [cos(θ)]
        lmax = 20
        coeffs = spherical_harmonic_coefficients(f_costheta, lmax, 1)
        
        # ℓ=1 is row 2, m=0 is column 1
        @test coeffs[2, 1, 1] ≈ sqrt(4π/3)
        
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs[:, :, 1])
        coeffs_copy[2, 1] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

    @testset "P₂(cos θ) ~ Y₂₀" begin
        # (3cos²(θ) - 1)/2 = P₂(cos θ) = √(4π/5) Y₂₀
        # Y₂₀ means ℓ=2, m=0
        f_P2(ϕ, θ) = [(3 * cos(θ)^2 - 1) / 2]
        lmax = 20
        coeffs = spherical_harmonic_coefficients(f_P2, lmax, 1)
        
        # ℓ=2 is row 3, m=0 is column 1
        @test coeffs[3, 1, 1] ≈ sqrt(4π/5)
        
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs[:, :, 1])
        coeffs_copy[3, 1] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

    @testset "sin(θ)sin(ϕ) ~ Y₁⁻¹ (real SH), sin(θ)cos(ϕ) ~ Y₁¹ (real SH)" begin
        # In real spherical harmonics: sin(θ)sin(ϕ) = √(4π/3) Y₁⁻¹
        f_y(ϕ, θ) = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ)]
        lmax = 20
        coeffs = spherical_harmonic_coefficients(f_y, lmax, 2)
        
        @test coeffs[1, 3, 1] ≈ sqrt(4π/3)    # Y₁¹ (real)
        @test coeffs[1, 2, 2] ≈ sqrt(4π/3)    # Y₁⁻¹ (real)
        
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs)
        coeffs_copy[1, 3, 1] = 0
        coeffs_copy[1, 2, 2] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

    @testset "SHPlan Interface Consistency Tests" begin
        
        # Shared parameters
        lmax = 20
        
        # Helper to manually compute integral from raw coefficients
        function manual_integration(coeffs)
            # The integral is the monopole moment (Y₀₀) scaled by √4π
            # In FastTransforms layout, Y₀₀ is at [1, 1]
            return coeffs[1, 1, :] * sqrt(4π)
        end

        @testset "Scalar Function: f = 1 + cos(θ)^2" begin
            K = 1
            plan = SHPlan(lmax, K)
            
            # Define function
            f(ϕ, θ) = [1.0 + cos(θ)^2]
            
            # 1. New API Method
            res_api = integrate_angular(plan, f)
            
            # 2. Direct Method (No Plan)
            raw_coeffs = spherical_harmonic_coefficients(f, lmax, K)
            res_manual = manual_integration(raw_coeffs)
            
            @test res_api ≈ res_manual atol=1e-12
        end

        @testset "Vector Function: f = [sin(θ)cos(ϕ), cos(θ)]" begin
            K = 2
            plan = SHPlan(lmax, K)
            
            # Define vector-valued function
            f_vec(ϕ, θ) = [sin(θ)*cos(ϕ), cos(θ)]
            
            # 1. New API Method
            res_api = integrate_angular(plan, f_vec)
            
            # 2. Direct Method (No Plan)
            raw_coeffs = spherical_harmonic_coefficients(f_vec, lmax, K)
            res_manual = manual_integration(raw_coeffs)
            
            @test res_api ≈ res_manual atol=1e-12
        end
        
        @testset "High Frequency: f = cos(10θ)" begin
            # Test consistency even when resolution might be borderline
            # (Consistency should hold regardless of accuracy)
            K = 1
            lmax_low = 10 # Just barely enough to capture basic modes?
            plan = SHPlan(lmax_low, K)
            
            f_high(ϕ, θ) = [cos(5*θ)] # Lower freq to ensure fit in lmax=10
            
            # 1. New API Method
            res_api = integrate_angular(plan, f_high)
            
            # 2. Direct Method
            raw_coeffs = spherical_harmonic_coefficients(f_high, lmax_low, K)
            res_manual = manual_integration(raw_coeffs)
            
            @test res_api ≈ res_manual atol=1e-12
        end

    end

    @testset "Exponential Function: f = exp(sin(θ)cos(ϕ))" begin
        # This function requires an infinite number of SH modes to represent exactly,
        # but it has a clean analytical integral: 4π * sinh(1).
        
        # 1. Define the function (returns vector as per API)
        # Note: sin(θ)cos(ϕ) is the Cartesian x-coordinate on the sphere
        f_exp(ϕ, θ) = [exp(sin(θ) * cos(ϕ))]
        
        # 2. Define analytical expected value
        expected_val = 4π * sinh(1.0)
        
        # 3. Compute using SHPlan
        # We use a sufficiently high lmax because the series is infinite (though converges fast)
        lmax = 30
        K = 1
        plan = SHPlan(lmax, K)
        
        # integrate_angular returns a Vector, so we extract the first element
        computed_val = integrate_angular(plan, f_exp)[1]
        
        # 4. Verify
        # We use a real comparison since the imaginary part should be negligible/zero
        @test real(computed_val) ≈ expected_val atol=1e-12
        @test abs(imag(computed_val)) < 1e-12
    end
end
