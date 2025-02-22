begin
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
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

function mc_surface_area(bubbles:: Bubbles, N:: Int):: Measurement
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

end

begin
    R = 2.
    bubbles = Vector{Bubble}([Bubble(Point3(0., 0., 0.), R)])    
    @test mc_surface_area(bubbles, 1_000)[1] ≈ 4π * (R ^ 2)
    _Δ = Δ(1_000)
    @test  surface_area(bubbles, _Δ) ≈ (4π * (R ^ 2))
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
    Ns = 4 .^ (3:9)
    sa2s = Vector{Measurement{Float64}}([])
    for N in Ns
        push!(sa2s, mc_surface_area(bubbles, N))
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10)
    scatter!(ax, Ns, value.(sa2s))
    errorbars!(ax, Ns, value.(sa2s), uncertainty.(sa2s))
    hlines!(ax, [sa1], linestyle=:dash, color="black")
    fig
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
    Ns = 4 .^ (3:10)
    sa2s = Vector{Measurement{Float64}}([])
    for N in Ns
        push!(sa2s, mc_surface_area(bubbles, N))
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10)
    scatter!(ax, Ns, value.(sa2s))
    errorbars!(ax, Ns, value.(sa2s), uncertainty.(sa2s))
    hlines!(ax, [sa1], linestyle=:dash, color="black")
    fig
end

begin
    snapshots = load("evolution_ensemble.jld2", "snapshots")
    bubbles = current_bubbles.(snapshots)[1]
    _Δ = Δ(length(bubbles))
    sa1 = surface_area(bubbles, _Δ)
    Ns = 4 .^ (3:10)
    sa2s = Vector{Measurement{Float64}}([])
    for N in Ns
        push!(sa2s, mc_surface_area(bubbles, N))
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10)
    scatter!(ax, Ns, value.(sa2s))
    errorbars!(ax, Ns, value.(sa2s), uncertainty.(sa2s))
    hlines!(ax, [sa1], linestyle=:dash, color="black")
    fig
end

function mc_bubble_area(bubble_index:: Int, bubbles:: Bubbles, N:: Int)
    bubble = bubbles[bubble_index]
    μ_dist = Uniform(-1., 1.)
    ϕ_dist = Uniform(0., 2π)
    μs = rand(μ_dist, N)
    ϕs = rand(ϕ_dist, N)
    ps = map(zip(μs, ϕs)) do (μ, ϕ)
        s = sqrt(1 - μ ^ 2)
        bubble.center + bubble.radius * Vec3(s * cos(ϕ), s * sin(ϕ), μ)
    end
    tot_inside = 0
    for p in ps
        inside_others = false
        for (nb, bubble) in enumerate(bubbles)
            nb == bubble_index && continue
            p ∈ bubble && (inside_others=true; break)
        end
        inside_others && continue
        tot_inside += 1
    end
    _mean = (tot_inside / N)
    _std = sqrt(tot_inside * ((1 - _mean) ^ 2) + (N - tot_inside) * (_mean ^ 2))/sqrt(N * (N - 1))
    return (_mean ± _std) * (4π * (bubble.radius ^ 2))
end

begin
    snapshots = load("evolution_ensemble.jld2", "snapshots")
    bubbles = current_bubbles.(snapshots)[1]
    domes = intersection_domes(bubbles)
    _Δ = Δ(length(bubbles))
    b = 1
    sa1 = bubble_surface_area(bubbles[b], domes[b], _Δ)
    Ns = 4 .^ (3:10)
    sa2s = Vector{Measurement{Float64}}([])
    for N in Ns
        push!(sa2s, mc_bubble_area(b, bubbles, N))
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xscale=log10)
    scatter!(ax, Ns, value.(sa2s))
    errorbars!(ax, Ns, value.(sa2s), uncertainty.(sa2s))
    hlines!(ax, [sa1], linestyle=:dash, color="black")
    fig
end

# begin  # Compare that total and bubble wise montecarlo esimates are consistent
#     Ns = 4 .^ (3:8)
#     sa2s1 = mc_surface_area.((bubbles, ), Ns)
#     sa2s2 = [sum(mc_bubble_area(b, bubbles, N) for b in 1:length(bubbles)) for N in Ns]
#     fig = Figure()
#     ax = Axis(fig[1, 1], xscale=log10)
#     scatter!(ax, Ns, value.(sa2s1), label="total")
#     errorbars!(ax, Ns, value.(sa2s1), uncertainty.(sa2s1))
#     scatter!(ax, Ns, value.(sa2s2), label="bubble wise")
#     errorbars!(ax, Ns, value.(sa2s2), uncertainty.(sa2s1))
#     axislegend(ax)
#     fig
# end

function mc_Δ(μ:: Float64, bubble_index:: Int, bubbles:: Bubbles, N:: Int)
    bubble = bubbles[bubble_index]
    ϕs = rand(Uniform(0., 2π), N)
    s = sqrt(1 - μ ^ 2)
    ps = map(ϕs) do ϕ
        bubble.center + bubble.radius * Vec3(s * cos(ϕ), s * sin(ϕ), μ)
    end
    tot_inside = 0
    for p in ps
        inside_others = false
        for (nb, bubble) in enumerate(bubbles)
            nb == bubble_index && continue
            p ∈ bubble && (inside_others=true; break)
        end
        inside_others && continue
        tot_inside += 1
    end
    _mean = (tot_inside / N)
    _std = sqrt(tot_inside * ((1 - _mean) ^ 2) + (N - tot_inside) * (_mean ^ 2))/sqrt(N * (N - 1))
    return (_mean ± _std) * 2π
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    μs = -1.:0.01:1.
    n_b = 1
    lines!(ax, μs, _Δ.(μs, (bubbles[n_b], ), (domes[n_b], )), label="Geometrical")
    N = 1_000
    measurements = mc_Δ.(μs, (n_b, ), (bubbles, ), (N, ))
    scatter!(ax, μs, value.(measurements), label="MC", color="purple")
    errorbars!(ax, μs, value.(measurements), uncertainty.(measurements), color="purple")
    # vlines!(polar_limits(bubbles[2].radius, domes[2]), linestyle=:dash)
    axislegend(ax, position=:lt)
    fig
end