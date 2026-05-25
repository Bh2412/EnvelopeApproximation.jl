"""
Unit tests for RayCastingLightConesSurface: collision search and spherical quadrature.
"""

using EnvelopeApproximation
using EnvelopeApproximation.RayCastingLightConesSurface
import EnvelopeApproximation.RayCastingLightConesSurface: collision_time, prepare_source!
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using Test
using StaticArrays
using LinearAlgebra

@testset "RayCastingLightConesSurface" begin

    # ═════════════════════════════════════════════════════════════════════════════
    # Collision Time Computation
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
    # Vacuum Boundary Conditions
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "Vacuum boundary stop time" begin
        v = 1.0
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))  # box [-5, 5]^3

        # Source at origin, ray along +x: boundary at x=5, τ_boundary = 5/v
        source = LightConeSource(1, 0.0, SVector(0.0, 0.0, 0.0))
        n̂_px = SVector(1.0, 0.0, 0.0)
        @test isapprox(boundary_stop_time(Vacuum(), space, source, n̂_px, v), 5.0, rtol=1e-12)

        # Ray along -x: boundary at x=-5, τ_boundary = 5/v
        n̂_mx = SVector(-1.0, 0.0, 0.0)
        @test isapprox(boundary_stop_time(Vacuum(), space, source, n̂_mx, v), 5.0, rtol=1e-12)

        # Source offset to (2,0,0): ray along +x hits x=5 in distance 3 → τ = 3/v
        source_offset = LightConeSource(1, 0.0, SVector(2.0, 0.0, 0.0))
        @test isapprox(boundary_stop_time(Vacuum(), space, source_offset, n̂_px, v), 3.0, rtol=1e-12)

        # Periodic: always Inf
        @test boundary_stop_time(Periodic(), space, source, n̂_px, v) == Inf
    end

    @testset "ray_stop_time: collision vs boundary in vacuum" begin
        v = 1.0

        # Source at (0,0,0) and blocker at (5,0,0), both nucleated at t=0.
        # Collision time along n̂=(1,0,0): τ_collision = 2.5.
        nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        nuc2 = (time=0.0, site=Point3(5.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc1, nuc2], 1000.0)
        source = LightConeSource(1, 0.0, SVector(0.0, 0.0, 0.0))
        n̂ = SVector(1.0, 0.0, 0.0)

        # Large box (L=100): boundary at x=50, τ_boundary=50. Collision (τ=2.5) wins.
        ctx_large = build_lightcone_context(snapshot, BoxSpace(100.0, Point3(0.0, 0.0, 0.0)), Vacuum(); v=v)
        prepare_source!(ctx_large, source)
        @test isapprox(ray_stop_time(ctx_large, source, n̂), 2.5, rtol=1e-10)

        # Small box (L=3): boundary at x=1.5, τ_boundary=1.5. Boundary wins over collision.
        ctx_small = build_lightcone_context(snapshot, BoxSpace(3.0, Point3(0.0, 0.0, 0.0)), Vacuum(); v=v)
        prepare_source!(ctx_small, source)
        @test isapprox(ray_stop_time(ctx_small, source, n̂), 1.5, rtol=1e-10)
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Spherical Quadrature Scheme
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

end
