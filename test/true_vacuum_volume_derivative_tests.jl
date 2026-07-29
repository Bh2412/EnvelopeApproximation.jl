using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.RayCastingEnvelopeIntegration
using StaticArrays
using Test

@testset "TrueVacuumVolumeDerivative kernel" begin
    @test_throws ArgumentError TrueVacuumVolumeDerivative([1.0, 0.0])
    @test_throws ArgumentError TrueVacuumVolumeDerivative([0.0, 1.0]; v=0.0)

    times = [-0.5, 0.0, 0.5, 1.0]
    @test TrueVacuumVolumeDerivative(times).v == 1.0

    v = 0.5
    kernel = TrueVacuumVolumeDerivative(times; v=v)
    derivative = allocate_accumulant(kernel)
    source = EnvelopeSource(1, 0.0, SVector(0.0, 0.0, 0.0))

    accumulate_ray!(
        derivative, kernel, source, SVector(1.0, 0.0, 0.0), 2.0, 0.75,
    )

    @test derivative ≈ [0.0, 0.0, 2.0 * v^3 * 0.5^2, 0.0]

    snapshot = BubblesSnapShot(
        [(time=0.0, site=Point3(0.0, 0.0, 0.0))],
        1.0,
    )
    space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
    boundary_condition = Periodic()
    integration_times = [0.0, 0.5, 1.0]

    context = build_envelope_context(
        snapshot,
        space,
        boundary_condition;
        v=v,
    )
    sources = envelope_sources(snapshot)
    markers = get_markers(UniformSphericalCapScheme(4, 8))
    integrated = only(envelope_integral(
        (TrueVacuumVolumeDerivative(integration_times; v=v),),
        sources,
        context,
        markers,
    ))

    @test integrated ≈ 4π * v^3 .* integration_times.^2

    @testset "two equal bubbles" begin
        separation = 1.0
        two_bubble_v = 0.8
        contact_time = separation / (2 * two_bubble_v)
        two_bubble_times = [0.0, 0.25, contact_time, 1.0, 1.25]
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
            boundary_condition;
            v=two_bubble_v,
        )
        two_bubble_sources = envelope_sources(two_bubble_snapshot)

        # The equal-sphere overlap volume is credited to Eric W. Weisstein,
        # "Sphere-Sphere Intersection," MathWorld:
        # https://mathworld.wolfram.com/Sphere-SphereIntersection.html
        #
        # MathWorld gives V_overlap = π(4R+d)(2R-d)²/12. Differentiating
        # V_union = 2(4πR³/3) - V_overlap with R=vt gives the expressions below.
        function equal_sphere_union_volume_derivative(time, distance, wall_speed)
            radius = wall_speed * time
            if 2 * radius <= distance
                return 8π * wall_speed * radius^2
            end
            return 2π * wall_speed * radius * (2 * radius + distance)
        end

        # The geometry is azimuthally symmetric about z.
        two_bubble_markers = get_markers(UniformSphericalCapScheme(4096, 1))
        numerical_derivative = only(envelope_integral(
            (TrueVacuumVolumeDerivative(
                two_bubble_times;
                v=two_bubble_v,
            ),),
            two_bubble_sources,
            two_bubble_context,
            two_bubble_markers,
        ))
        analytic_derivative = map(two_bubble_times) do time
            equal_sphere_union_volume_derivative(
                time,
                separation,
                two_bubble_v,
            )
        end

        @test numerical_derivative ≈ analytic_derivative rtol=2.0e-6 atol=1.0e-10
    end
end
