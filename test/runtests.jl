using EnvelopeApproximation
using Test
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.SurfaceIntegration as SI
import EnvelopeApproximation.GravitationalPotentials as GP
using Meshes

@testset "BubblesEvolution.jl" begin
    using EnvelopeApproximation.BubblesEvolution
    bs = BallSpace(1., Point3(0., 0., 0.))
    eg = ExponentialGrowth(1., 1.)
    ts = (_, _, fv) -> fv < 0.1
    res = evolve(eg, bs, termination_strategy=ts)
end

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
    r = 2.
    bubbles = Bubbles([Point3(0., 0., 0.)], [r])
    @test SI.surface_integral(z -> 1., bubbles, 10, 10) ≈ 4π * (r^2)
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
    times = LinRange(0., π / 2, 1000) |> collect
    s_vals = -(frequency ^ 2) * cos.(frequency * times)
    s = Source(times, s_vals)
    numerical_solution = ode_solution(s)
    exact_solution = cos.(frequency * times) .- 1
    println(max(abs.(numerical_solution - exact_solution)...))
    @test isapprox(numerical_solution, exact_solution, atol=1e-4)
end
@testset "MultiDimensional cosine_source" begin
    frequencies = [2π / i for i in 2:17] |> x -> reshape(x, 1, 4, 4)
    times = LinRange(0., π / 2, 1000) |> collect |> x -> reshape(x, :, 1, 1)
    s_vals = @. -(frequencies ^ 2) * cos(frequencies * times)
    s = Source(times |> x -> reshape(x, :), s_vals)
    numerical_solution = ode_solution(s)
    exact_solution = @. cos(frequencies * times) - 1
    println(max(abs.(numerical_solution - exact_solution)...))
    @test isapprox(numerical_solution, exact_solution, atol=1e-3)
end
end

end

@testset "StressEnergyTensor" begin
    import Meshes.+
    using EnvelopeApproximation
    import EnvelopeApproximation.BubbleBasics: Bubble, Bubbles
    import EnvelopeApproximation.SurfaceIntegration: BubbleSection
    import EnvelopeApproximation.StressEnergyTensor: surface_integral, td_integrand, _exp, volume_integral, T_ij
    using Plots
    
    R = 3.
    bubble_center = Point3(0., 0., 1.)
    bubbles = Bubbles([Bubble(bubble_center, R)])
    μs = LinRange(-1., 1., 100)
    xx_td_integrand = td_integrand((:x, :x), bubbles)
    zz_td_integrand = td_integrand((:z, :z), bubbles)
    ps = [BubbleSection(bubble_center + Point3(R * (1 - μ ^ 2) ^ (1/2), 0., R * μ), 1) for μ in LinRange(-1., 1., 100)]
    @test xx_td_integrand.(ps) ≈ 1 .- μs .^ 2
    @test zz_td_integrand.(ps) ≈ μs .^ 2 
    ks = [Point3(0., 0., z) for z in LinRange(0., 10., 11)]
    tensor_directions = [:trace, (:x, :x), (:y, :y), (:z, :z)]
    si = surface_integral(ks, bubbles, tensor_directions, 10, 10)
    @test size(si) == (length(ks), length(tensor_directions))
    @test reshape(si[:, 1], length(ks)) ≈ sum(si[:, 2:4], dims=2) |> x -> reshape(x, length(ks))
    vi = volume_integral(ks, bubbles, 10, 10, 10)
    T = T_ij(ks, bubbles, 10, 10, 10)
    print(T)
end;

