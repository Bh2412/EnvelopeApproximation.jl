using EnvelopeApproximation
using Test
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.SurfaceTesselation as SI
import EnvelopeApproximation.GravitationalPotentials as GP
using Meshes

@testset "GeometricStressEnergyTensor.jl" begin
@testset "ring_dome_intersection" begin
@testset "benchmark1" begin
    import EnvelopeApproximation.GeometricStressEnergyTensor: ∩, intersection_arcs, Δϕ′, ∥, ẑ, mod2π, align_ẑ, symmetric_tensor_inverse_rotation, SymmetricTensorMapping, surface_integral, ∫_ϕ, upper_right
    import EnvelopeApproximation.GeometricStressEnergyTensor: BubbleArcSurfaceIntegrand, fourier_mode, SphericalIntegrand
    import EnvelopeApproximation.GeometricStressEnergyTensor: potential_integral, BubbleArcPotentialIntegrand
    import EnvelopeApproximation.GeometricStressEnergyTensor: T_ij, ring_dome_intersection, IntervalSet, PeriodicInterval, complement, IntersectionArc, atan2π
    import EnvelopeApproximation.GeometricStressEnergyTensor: k̂ik̂jTij, ŋ_source
    R, n̂ , h = 1., (1. / √(2)) * Vec3(1., 0., 1.), 0.5
    x(μ) = 1. / √(2) - μ
    y2(μ) = (1. / 2 + √(2) * μ - 2 * μ^2)
    y(μ) = √(y2(μ))
    μp, μm = begin
        a, b = 1/√(8), √(6) / 4
        a - b, a + b
    end 
    ϕp(μ) = begin 
        if μ < μp
            return 0.
        elseif μ > μm
            return 0.
        else
            return atan(-y(μ), x(μ))
        end
    end
    Δ(μ) = begin
        if μ < μp
            return 0.
        elseif μ > μm
            return 2π
        else
            return atan(y(μ), x(μ)) - ϕp(μ)
        end
    end
    μs = -1.:0.0001:1.
    expected_ϕs = ϕp.(μs) .|> mod2π
    expected_Δs = Δ.(μs)
    pis = ring_dome_intersection.(μs, (R, ), (n̂, ), h)
    derived_ϕs = pis .|> x -> x.ϕ1
    derived_Δs = pis .|> x -> x.Δ
    @test all(isapprox.(derived_ϕs, expected_ϕs, atol=1e-8))
    @test all(isapprox.(derived_Δs, expected_Δs, atol=1e-8))
end
end
end

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
    import EnvelopeApproximation.BubblesIntegration.SurfaceTesselation: BubbleSection, Section, surface_sections
    import EnvelopeApproximation.StressEnergyTensor: coordinate_transformation
    using StaticArrays
    using Plots
    
    R = 3.
    bubble_center = Point3(0., 0., 1.)
    bubbles = Bubbles([Bubble(bubble_center, R)])
    sections = surface_sections(2, 2, bubbles)
    @show coordinate_transformation(SVector{2, Float64}(0., 0.), sections[1])
end;

