using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using Test
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration as SI
import EnvelopeApproximation.BubblesIntegration.VolumeIntegration as VI

@testset "BubblesIntegration.jl" begin
    

@testset "SurfaceIntegration.jl" begin
@testset "Unit Sphere Tesselation" begin
    ps = SI.unit_sphere_points(Float64(2π), 1.)
    @test all((ps .≈ [Point3(-sqrt(3. / 4), 0., -1. / 2), Point3(-sqrt(3. /4), 0., 1. / 2)]))
    ϕs, μs = SI.unit_sphere_tesselation(Float64(π), 2.)
    @test ϕs ≈ [π / 2, 3π / 2]
    @test μs ≈ [0.]
    ps = SI.unit_sphere_points(Float64(π), 2.)
    @test length(ps) == 2
    @test all(ps .≈ [Point3(0., 1. , 0.), Point3(0., -1., 0.)])
end
    bubbles = Bubbles([Point3(0., 0., 0.), Point3(0., 0., 1.)], [1., 2.])
    @test length(bubbles) == 2
    usps = SI.unit_sphere_points(Float64(2π), 1.)
    psps = SI._preliminary_surface_points(usps, bubbles)
    @test size(psps) == (length(usps), 2)
    @test all(psps[:, 1] .≈ [Point3(-sqrt(3. / 4), 0., -1. / 2), Point3(-sqrt(3. /4), 0., 1. / 2)])
    @test all(psps[:, 2] .≈ [Point3(-sqrt(3.), 0., -0.), Point3(-sqrt(3.), 0., 2.)])
end
end


@testset "VolumeIntegration" begin
    _ns = VI.n(1. / 3, 1. / 6)
    @test _ns == 2
    ubt = VI.unit_ball_tesselation(1. / 6, Float64(2π), 2.)
    @test ubt[1] == [1. / 12, 1. / 4]
    bubbles = Bubbles([Point3(0., 0., 0.)], [1.])
    VI.volume_integral(x -> 1., bubbles, 1. /6, Float64(2π), 2.)
end
