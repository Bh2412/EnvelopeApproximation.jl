using Test
using EnvelopeApproximation.SphericalHarmonics

@testset "Spherical Harmonic Transform Tests" begin

    @testset "Constant function" begin
        # f = 1 should give only Y₀₀
        f_const(ϕ, θ) = [1.0]
        coeffs = spherical_harmonic_coefficients(f_const, 20, 1)
        
        # Y₀₀ should be non-zero
        @test abs(coeffs[1, 1, 1]) ≈ sqrt(4π)
        println("Y₀₀ coefficient: ", coeffs[1, 1, 1])
        println("Expected: ", sqrt(4π))
        
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

    @testset "sin(θ)cos(ϕ) ~ Y₁¹ (real SH)" begin
        # In real spherical harmonics: sin(θ)cos(ϕ) = √(4π/3) Y₁¹
        f_x(ϕ, θ) = [sin(θ) * cos(ϕ)]
        lmax = 1
        coeffs = spherical_harmonic_coefficients(f_x, lmax, 1)
        
        @test coeffs[1, 3, 1] ≈ sqrt(4π/3)    # Y₁¹ (real)
        
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs[:, :, 1])
        coeffs_copy[1, 3] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

    @testset "sin(θ)sin(ϕ) ~ Y₁⁻¹ (real SH)" begin
        # In real spherical harmonics: sin(θ)sin(ϕ) = √(4π/3) Y₁⁻¹
        f_y(ϕ, θ) = [sin(θ) * sin(ϕ)]
        lmax = 20
        coeffs = spherical_harmonic_coefficients(f_y, lmax, 1)
        
        @test coeffs[1, 2, 1] ≈ sqrt(4π/3)    # Y₁⁻¹ (real)
        
        # Check all other coefficients are zero
        coeffs_copy = copy(coeffs[:, :, 1])
        coeffs_copy[1, 2] = 0
        @test maximum(abs.(coeffs_copy)) < 1e-10
    end

end