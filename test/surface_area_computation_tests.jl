begin
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    import EnvelopeApproximation.BubblesEvolution: BallSpace
    using EnvelopeApproximation.GeometricStressEnergyTensor
    import EnvelopeApproximation.GeometricStressEnergyTensor: Δ, intersection_domes, IntersectionDome, polar_limits
    using Distributions
    using QuadGK
    using Test
    using Statistics
    import Base.rand
    using JLD2
    using Measurements
    import Measurements: value, uncertainty
    import Base.∈
    using LinearAlgebra
end

begin

function bubble_surface_area(bubble:: Bubble, domes:: Vector{IntersectionDome}, _Δ:: Δ; kwargs...)
    _polar_limits = polar_limits(bubble.radius, domes)
    return quadgk(μ -> bubble.radius ^ 2 * _Δ(μ, bubble, domes), _polar_limits...)[1]
end

function surface_area(bubbles:: Bubbles, _Δ:: Δ; kwargs...):: Float64
    domes = intersection_domes(bubbles)
    _surface_area = 0.
    for (i, bubble) in enumerate(bubbles)
        _surface_area += bubble_surface_area(bubble, domes[i], _Δ; kwargs...)
    end
    return _surface_area
end

function surface_area(bubbles:: Bubbles, ball_space:: BallSpace, _Δ:: Δ; kwargs...):: Float64
    domes = intersection_domes(bubbles, ball_space)
    _surface_area = 0.
    for (i, bubble) in enumerate(bubbles)
        _surface_area += bubble_surface_area(bubble, domes[i], _Δ; kwargs...)
    end
    return _surface_area
end


function mc_surface_area(bubbles:: Bubbles, N:: Int):: Measurement{Float64}
    surface_areas = map(b -> 4π * ((b.radius) ^ 2), bubbles)
    uncollieded_surface_area = sum(surface_areas)
    bubble_dist = Categorical(surface_areas ./ uncollieded_surface_area)
    μ_dist = Uniform(-1., 1.)
    ϕ_dist = Uniform(0., 2π)
    bubble_indices = rand(bubble_dist, N)
    μs = rand(μ_dist, N)
    ϕs = rand(ϕ_dist, N)
    ps = map(zip(bubble_indices, μs, ϕs)) do (b, μ, ϕ)
        s = sqrt(1 - μ ^ 2)
        bubble = bubbles[b]
        bubble.center + bubble.radius * Vec3(s * cos(ϕ), s * sin(ϕ), μ)
    end
    tot_inside = 0
    for (b, p) in zip(bubble_indices, ps)
        inside_others = false
        for (nb, bubble) in enumerate(bubbles)
            nb == b && continue
            p ∈ bubble && (inside_others=true; break)
        end
        inside_others && continue
        tot_inside += 1
    end
    _mean = (tot_inside / N)
    _std = sqrt(tot_inside * ((1 - _mean) ^ 2) + (N - tot_inside) * (_mean ^ 2))/sqrt(N * (N - 1))
    return (_mean ± _std) * uncollieded_surface_area
end

function mc_surface_area(bubbles:: Bubbles, ball_space:: BallSpace, N:: Int):: Measurement{Float64}
    surface_areas = map(b -> 4π * ((b.radius) ^ 2), bubbles)
    uncollieded_surface_area = sum(surface_areas)
    bubble_dist = Categorical(surface_areas ./ uncollieded_surface_area)
    μ_dist = Uniform(-1., 1.)
    ϕ_dist = Uniform(0., 2π)
    bubble_indices = rand(bubble_dist, N)
    μs = rand(μ_dist, N)
    ϕs = rand(ϕ_dist, N)
    ps = map(zip(bubble_indices, μs, ϕs)) do (b, μ, ϕ)
        s = sqrt(1 - μ ^ 2)
        bubble = bubbles[b]
        bubble.center + bubble.radius * Vec3(s * cos(ϕ), s * sin(ϕ), μ)
    end
    tot_inside = 0
    for (b, p) in zip(bubble_indices, ps)
        inside_others = false
        p ∉ ball_space && continue
        for (nb, bubble) in enumerate(bubbles)
            nb == b && continue
            p ∈ bubble && (inside_others=true; break)
        end
        inside_others && continue
        tot_inside += 1
    end
    _mean = (tot_inside / N)
    _std = sqrt(tot_inside * ((1 - _mean) ^ 2) + (N - tot_inside) * (_mean ^ 2))/sqrt(N * (N - 1))
    return (_mean ± _std) * uncollieded_surface_area
end

end

@testset "Surface Area Integration" begin
begin
    R = 2.
    bubbles = Vector{Bubble}([Bubble(Point3(0., 0., 0.), R)])    
    _Δ = Δ(1_000)
    @testset "single bubble" begin
    @test mc_surface_area(bubbles, 1_000)[1] ≈ 4π * (R ^ 2)
    @test  surface_area(bubbles, _Δ) ≈ (4π * (R ^ 2))
    end
end

begin
    using CairoMakie
    R = 4.8
    d = 1.2
    nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
    snapshot = BubblesSnapShot(nucleations, R)
    bubbles = current_bubbles(snapshot)
    _Δ = Δ(2)
    sa1 = surface_area(bubbles, _Δ)
    N = 4 ^ 8
    sa2 = mc_surface_area(bubbles, N)
    @assert (uncertainty(sa2) / value(sa2)) <= 0.01
    @testset "even 2 bubbles" begin
    @test abs(value(sa2) - sa1) < (5 * uncertainty(sa2))
    end
end

begin
    using CairoMakie
    R = 4.8
    d = 1.2
    nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=1., site=Point3(0., 0., d / 2))]
    snapshot = BubblesSnapShot(nucleations, R)
    bubbles = current_bubbles(snapshot)
    _Δ = Δ(2)
    sa1 = surface_area(bubbles, _Δ)
    N = 4 ^ 8
    sa2 = mc_surface_area(bubbles, N)
    @assert (uncertainty(sa2) / value(sa2)) <= 0.01
    @testset "uneven 2 bubbles" begin
    @test abs(value(sa2) - sa1) < (5 * uncertainty(sa2))
    end
end

begin
    data_file = joinpath(@__DIR__, "test_data", "evolution_ensemble.jld2")
    snapshots = load(data_file, "snapshots")[1:100]
    bubbless = current_bubbles.(snapshots)
    _Δ = Δ(max(length.(bubbless)...))
    N = 4 ^ 8
    @testset "General Ensemble" begin
    for bubbles in bubbless
        sa1 = surface_area(bubbles, _Δ)
        sa2 = mc_surface_area(bubbles, N)
        @assert (uncertainty(sa2) / value(sa2)) <= 0.01
        @test abs(value(sa2) - sa1) < (5 * uncertainty(sa2))
    end
    end
end

begin
    data_file = joinpath(@__DIR__, "test_data", "evolution_ensemble.jld2")
    ball_space = load(data_file, "ball_space")
    snapshots = load(data_file, "snapshots")[1:100]
    bubbless = current_bubbles.(snapshots)
    _Δ = Δ(max(length.(bubbless)...))
    N = 4^9
    @testset "General Ensemble With Reflective Boundary Conditions" begin
    for bubbles in bubbless
        sa1 = surface_area(bubbles, ball_space, _Δ)
        @assert sa1 < surface_area(bubbles, _Δ)
        sa2 = mc_surface_area(bubbles, ball_space, N)
        @assert (uncertainty(sa2) / value(sa2)) <= 0.01
        @test abs(value(sa2) - sa1) < (5 * uncertainty(sa2))
    end
    end
end

end
