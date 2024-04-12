using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using Test
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration as SI
import EnvelopeApproximation.BubblesIntegration.VolumeIntegration as VI
import EnvelopeApproximation.GravitationalPotentials as GP

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

@testset "GravitationalPotentials" begin

@testset "SecondOrderODESolver" begin
using EnvelopeApproximation.GravitationalPotentials.SecondOrderODESolver
@testset "simple_delta_source" begin
    n = 11
    times = collect(LinRange(0., n-1, n))
    values = zeros(n)
    values[1] = 2.
    source = Source(times, values)
    # equivalent to a constant source of amplitude 1. between t=0. and t=1.
    numerical_solution = ode_solution(source)
    exact_solution = begin
        a = 0.
        b = 1.
        sol = (1/2) * (b-a) * (2 * times .- (a + b))
        sol[1] = 0.
        sol
    end
    @test numerical_solution == exact_solution
end
@testset "cosine_source" begin
    frequency = 2π / 5
    times = LinRange(0., π / 2, 1000)
    s_vals = -(frequency ^ 2) * cos.(frequency * times)
    s = Source(times, s_vals)
    numerical_solution = ode_solution(s)
    exact_solution = cos.(frequency * times) .- 1
    @test isapprox(numerical_solution, exact_solution, atol=1e-4)
end
end


end
