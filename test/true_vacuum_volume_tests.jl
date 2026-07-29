using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.RayCastingEnvelopeIntegration
using StaticArrays
using Test

@testset "TrueVacuumVolume kernel" begin
    @test_throws ArgumentError TrueVacuumVolume([1.0, 0.0])
    @test_throws ArgumentError TrueVacuumVolume([0.0, 1.0]; v=0.0)

    times = [-0.5, 0.0, 0.5, 1.0]
    @test TrueVacuumVolume(times).v == 1.0

    v = 0.5
    kernel = TrueVacuumVolume(times; v=v)
    volumes = allocate_accumulant(kernel)
    source = EnvelopeSource(1, 0.0, SVector(0.0, 0.0, 0.0))

    accumulate_ray!(
        volumes, kernel, source, SVector(1.0, 0.0, 0.0), 2.0, 0.75,
    )

    @test volumes ≈ [
        0.0,
        0.0,
        2.0 * (v * 0.5)^3 / 3.0,
        2.0 * (v * 0.75)^3 / 3.0,
    ]

    snapshot = BubblesSnapShot(
        [(time=0.0, site=Point3(0.0, 0.0, 0.0))],
        1.0,
    )
    space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
    boundary_condition = Periodic()
    quadrature = UniformSphericalCapScheme(4, 8)
    integration_times = [0.0, 0.5, 1.0]

    context = build_envelope_context(snapshot, space, boundary_condition)
    sources = envelope_sources(snapshot)
    markers = get_markers(quadrature)
    integrated = only(envelope_integral(
        (TrueVacuumVolume(integration_times),),
        sources,
        context,
        markers,
    ))

    @test integrated ≈ 4π / 3.0 .* integration_times.^3

    @testset "two-bubble analytic volume" begin
        separation = 1.0
        two_bubble_times = [0.0, 0.25, 0.5, 0.75, 1.0]
        two_bubble_snapshot = BubblesSnapShot(
            [
                (time=0.0, site=Point3(0.0, 0.0, -separation / 2)),
                (time=0.0, site=Point3(0.0, 0.0,  separation / 2)),
            ],
            last(two_bubble_times),
        )
        two_bubble_space = BoxSpace(20.0, Point3(0.0, 0.0, 0.0))

        two_bubble_context = build_envelope_context(
            two_bubble_snapshot,
            two_bubble_space,
            boundary_condition,
        )
        two_bubble_sources = envelope_sources(two_bubble_snapshot)

        # The geometry is azimuthally symmetric about z, so one azimuthal
        # marker per polar ring performs the ϕ integral exactly.
        two_bubble_markers = get_markers(UniformSphericalCapScheme(4096, 1))
        numerical_volume = only(envelope_integral(
            (TrueVacuumVolume(two_bubble_times),),
            two_bubble_sources,
            two_bubble_context,
            two_bubble_markers,
        ))

        # See https://mathworld.wolfram.com/Sphere-SphereIntersection.html
        function equal_sphere_union_volume(radius, distance)
            sphere_volume = 4π * radius^3 / 3.0
            2 * radius <= distance && return 2 * sphere_volume

            overlap_volume =
                π * (4 * radius + distance) * (2 * radius - distance)^2 / 12.0
            return 2 * sphere_volume - overlap_volume
        end

        analytic_volume = map(two_bubble_times) do radius
            equal_sphere_union_volume(radius, separation)
        end

        @test numerical_volume ≈ analytic_volume rtol=2.0e-6 atol=1.0e-10
    end
end
