"""
Unit tests for ray-tracing T_ij computation.
"""

using EnvelopeApproximation
using EnvelopeApproximation.RayTracingStressEnergyTensor
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.TwoPointStressEnergyTensorModule: RayTracingCosineTimeIntegratedTwoPointStressEnergyTensor
using Test
using StaticArrays
using LinearAlgebra
using QuadGK

@testset "Ray Tracing Stress Energy Tensor" begin

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: I₃ Analytic Integral
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "I3 Integral Evaluation" begin
        # α = 0 case
        @test isapprox(I3(0.0, 0.0, 1.0), (1.0^4 - 0.0^4) / 4.0, rtol=1e-10)
        @test isapprox(imag(I3(0.0, 0.0, 1.0)), 0.0, atol=1e-14)

        # Near-zero α uses Taylor series; should match α = 0 closely
        @test isapprox(I3(1e-12, 0.0, 1.0), I3(0.0, 0.0, 1.0), rtol=1e-8)

        # Swap limits changes sign
        @test isapprox(I3(1.5, 0.5, 1.5), -I3(1.5, 1.5, 0.5), rtol=1e-10)

        # Correctness against numerical quadrature — this catches wrong signs in the
        # antiderivative formula.
        for (α, a, b) in [(2.0, 0.0, 1.0), (0.5, 0.3, 2.0), (-1.0, 0.0, 3.0), (5.0, 1.0, 2.0)]
            ref, _ = quadgk(τ -> τ^3 * cis(α * τ), a, b; rtol=1e-12)
            @test isapprox(I3(α, a, b), ref, rtol=1e-8) broken=false
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: Collision Time Computation
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "Collision Time Computation" begin
        v = 1.0

        # Non-degenerate head-on case:
        #   bubble i at origin nucleating at t=0
        #   bubble j at (3,0,0) nucleating at t=-1  (radius=1 at t=0, so no overlap)
        #   ray n̂=(1,0,0): wall of i meets wall of j at τ=1
        #     check: |vτ n̂ − (3,0,0)| = |(1,0,0)−(3,0,0)| = 2 = v(1+1) ✓
        center_i = SVector(0.0, 0.0, 0.0)
        center_j = SVector(3.0, 0.0, 0.0)
        t_i, t_j = 0.0, -1.0
        n̂ = SVector(1.0, 0.0, 0.0)

        τ = collision_time(center_i, center_j, n̂, t_i, t_j, v)
        @test τ !== nothing
        @test isapprox(τ, 1.0, rtol=1e-10)

        # Ray pointing away: no collision
        n̂_away = SVector(-1.0, 0.0, 0.0)
        @test collision_time(center_i, center_j, n̂_away, t_i, t_j, v) === nothing

        # Same-time nucleation: two bubbles at t=0, separated by 4 along x.
        # Wall elements along +x meet at τ=2 (each travels half the gap).
        center_k = SVector(4.0, 0.0, 0.0)
        τ2 = collision_time(center_i, center_k, SVector(1.0, 0.0, 0.0), 0.0, 0.0, v)
        @test τ2 !== nothing
        @test isapprox(τ2, 2.0, rtol=1e-10)

        # Later-nucleating bubble: bubble j nucleates at t=1 > t_i=0.
        # Collision must not be reported before t_j = t=1, i.e. τ ≥ 1.
        center_l = SVector(5.0, 0.0, 0.0)
        τ3 = collision_time(center_i, center_l, SVector(1.0, 0.0, 0.0), 0.0, 1.0, v)
        @test τ3 === nothing || τ3 >= 1.0 - 1e-10
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: Quadrature Scheme
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "UniformSphericalCapScheme" begin
        scheme = UniformSphericalCapScheme(3, 4)
        markers = get_markers(scheme)

        @test length(markers) == 12

        for marker in markers
            @test isapprox(norm(marker.n̂), 1.0, rtol=1e-10)
            @test marker.weight > 0.0
        end

        # Weights must sum to 4π (they are Δμ·Δϕ cells covering the full sphere)
        @test isapprox(sum(m.weight for m in markers), 4π, rtol=1e-10)
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: ray_T_ij returns a (A_plus, A_minus) tuple of correct shape
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "Single Bubble Ray-Tracing" begin
        nuc = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc], 1.0)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()

        scheme = UniformSphericalCapScheme(4, 8)
        strategy = RayTracingT_ij_CosineWeight(scheme)

        ks = [0.5, 1.0, 2.0]
        A_plus, A_minus = ray_T_ij(ks, snapshot, space, bc, strategy; ΔV=1.0)

        @test size(A_plus)  == (6, 3)
        @test size(A_minus) == (6, 3)

        # A_plus and A_minus must differ (they carry opposite phase e^{+ikt} vs e^{-ikt})
        @test !isapprox(A_plus, A_minus, rtol=1e-6)

        # Single bubble with no collisions: A_minus = conj(A_plus) because
        # the integrand for A− is the complex conjugate of A+ (α → −α reflects as well).
        @test isapprox(A_minus, conj.(A_plus), rtol=1e-6)
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: Two-point correlator is Hermitian and has positive diagonal
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "RayTracingCosineTimeIntegratedTwoPointTensor" begin
        nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        nuc2 = (time=0.2, site=Point3(1.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc1, nuc2], 2.0)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()

        scheme = UniformSphericalCapScheme(4, 8)
        strategy = RayTracingT_ij_CosineWeight(scheme)

        ks = [0.5, 1.0]
        result = RayTracingCosineTimeIntegratedTwoPointStressEnergyTensor(
            ks, snapshot, space, bc, strategy; ΔV=1.0
        )

        @test size(result) == (6, 6, 2)
        @test any(!iszero, result)

        for ki in 1:2
            M = result[:, :, ki]
            # Hermitian symmetry: M[i,j] = conj(M[j,i])
            @test isapprox(M, M', rtol=1e-8)
            # Diagonal entries are real non-negative (positive semi-definite diagonal)
            for ij in 1:6
                @test real(M[ij, ij]) >= -1e-12
            end
        end
    end

end  # @testset

println("All ray-tracing tests passed!")
