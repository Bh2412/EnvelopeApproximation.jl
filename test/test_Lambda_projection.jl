using Test
using LinearAlgebra
using StaticArrays
using EnvelopeApproximation
# Access the internal function Λ from the GravitationalWaves module
const GW = EnvelopeApproximation.GravitationalWaves.AzimuthalReductionBigpi
const Λ = GW.Λ

@testset "Λ (Lambda) Projection Tests" begin

    # Helper to create a 6-component vector from specific tensor elements
    # Indices: 1:xx, 2:xy, 3:xz, 4:yy, 5:yz, 6:zz
    function make_tensor(;xx=0, xy=0, xz=0, yy=0, yz=0, zz=0)
        return ComplexF64[xx, xy, xz, yy, yz, zz]
    end

    @testset "Positivity and Reality" begin
        # Test for random complex tensors
        for i in 1:100
            T = rand(ComplexF64, 6)
            val = Λ(T)
            
            # Check 1: Result is strictly real (imaginary part is 0)
            @test isapprox(imag(val), 0.0, atol=1e-12)
            
            # Check 2: Result is non-negative
            # Note: We use a small tolerance for floating point noise around 0
            @test real(val) >= -1e-14
        end
    end

    @testset "Analytical Cases" begin
        # Case 1: Pure Trace (Spherical symmetry)
        # T = diag(1, 1, 1) -> should project to 0 (no GWs)
        # Λ = 1/2 |Txx - Tyy|^2 + 2|Txy|^2
        T_trace = make_tensor(xx=1, yy=1, zz=1)
        @test Λ(T_trace) ≈ 0.0 atol=1e-12

        # Case 2: Pure + Polarization
        # T = diag(1, -1, 0)
        # Λ = 1/2 |1 - (-1)|^2 + 0 = 1/2 * 4 = 2
        T_plus = make_tensor(xx=1, yy=-1)
        @test Λ(T_plus) ≈ 2.0 atol=1e-12

        # Case 3: Pure x Polarization
        # Txy = 1
        # Λ = 0 + 2|1|^2 = 2
        T_cross = make_tensor(xy=1)
        @test Λ(T_cross) ≈ 2.0 atol=1e-12
        
        # Case 4: Mixed Polarization
        # T = diag(1, -1, 0) + Txy=1
        # Λ should be sum of individual powers: 2 + 2 = 4
        T_mixed = make_tensor(xx=1, yy=-1, xy=1)
        @test Λ(T_mixed) ≈ 4.0 atol=1e-12
    end

    @testset "Projection Property (Removal of non-GW terms)" begin
        # Any component involving z (xz, yz, zz) should not contribute 
        # to the final Λ value for a wave propagating in z.
        
        # Base tensor (Pure + polarization)
        T_base = make_tensor(xx=1, yy=-1)
        val_base = Λ(T_base) # Should be 2.0

        # Add junk in z components
        T_junk = make_tensor(xx=1, yy=-1, xz=5, yz=-3, zz=10)
        val_junk = Λ(T_junk)

        @test val_junk ≈ val_base atol=1e-12
    end

    @testset "Comparison with Direct Formula" begin
        # Verify against the explicit formula: 0.5*|xx-yy|^2 + 2*|xy|^2
        for i in 1:10
            T = rand(ComplexF64, 6)
            xx, xy, yy = T[1], T[2], T[4]
            
            expected = 0.5 * abs2(xx - yy) + 2.0 * abs2(xy)
            computed = Λ(T)
            
            @test computed ≈ expected atol=1e-12
        end
    end
end