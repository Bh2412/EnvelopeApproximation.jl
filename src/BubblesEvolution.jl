module BubblesEvolution

using EnvelopeApproximation.BubbleBasics
import Meshes.Point3
import Meshes.Vec3
import Random.AbstractRNG
using StatsBase
import StatsBase.sample
import Distances.Euclidean
import Distances.pairwise
import Base.∈
using Distributions
import Base.isless
using Intervals

Nucleation = NamedTuple{(:time, :site)}
isless(n1:: Nucleation, n2:: Nucleation) = isless(n1[:time], n2[:time])

struct BubblesSnapshot
    nucleations:: Vector{Nucleation}
    t:: Float64
    radial_profile:: Function
end

speed_of_light_profile(t:: Float64, c:: Float64 = 1.):: Float64 = c * t 

BubblesSnapShot() = BubblesSnapShot(Vector{Nucleation}(), 0., speed_of_light_profile)

function at_earlier_time(snap:: BubblesSnapshot, t:: Float64):: BubblesSnapshot
    nucleations = filter(nuc -> nuc[:time] <= t, snap.nucleations)
    return BubblesSnapshot(nucleations, t, snap.radial_profile)
end

function evolve(snap:: BubblesSnapshot, nucleations:: Vector{Nucleation}, Δt:: Float64)
    return BubblesSnapshot([snap.nucleations..., nucleations...], snap.t + Δt, snap.radial_profile)
end

function current_bubbles(snap:: BubblesSnapshot, t:: {Float64, Nothing} = nothing):: Bubbles
    if t ≡ nothing
        t = snap.t
    end
    return Bubbles([Bubble(nuc[:site], snap.radial_profile(t - nuc[:time])) for nuc in snap.nucleations])
end

abstract type AbstractSpace end

function sample(rng:: AbstractRNG, n:: Int64, space:: AbstractSpace):: Vector{Point3} 
    throw("Cant sample from abstract space $space")
end

struct BallSpace
    radius:: Float64
    center:: Point3
end

function unit_sphere_point(ϕ:: Float64, μ:: Float64):: Point3
    s = sqrt((1. - μ ^2))
    return Point3(s * cos(ϕ), s * sin(ϕ), μ)
end

function sample(rng:: AbstactRNG, n:: Int64, space:: BallSpace):: Vector{Point3}
    # r^3 is distributed uniformly over (0, 1)
    r = rand(rng, Uniform(0., 1.), n) .^ (1. / 3)
    # ϕ is distributed uniformly over (0, 2π)
    ϕ = rand(rng, Uniform(0., 2π) , n)
    # μ is distributed uniformly over (-1., 1.) 
    μ = rand(rng, Uniform(-1., 1.), n)
    v = begin
        s = (x -> sqrt(1 - x^2)).(μ)
        Vec3.(r .* s .* cos.(ϕ), r .* s .* sin.(ϕ), r .* μ)
    end
    return space.center .+ v
end
    
euclidean = Euclidean()
euc(point1:: Point3, point2:: Point3):: Float64 = euclidean(coordinates.([point1, point2])...)
∈(point:: Point3, bubble:: Bubble) :: Bool = euc(point, bubble.center) <= bubble.radius

function potential_sites(rng:: AbstractRNG, mean_nucleations:: Float64, space:: AbstractSpace)::
    n = rand(rng, Poisson(mean_nucleations))
    return sample(rng, n, space)
end

function fv_filter(existing_bubbles:: Bubbles):: Vector{Point3}
    return filter(s -> !any(s ∈ bubble for bubble in existing_bubbles))
end

function sample_nucleations(Δt:: Float64,
                            mean_nucleations:: Float64, 
                            space:: AbstractSpace,
                            existing_bubbles:: Bubbles,
                            t0:: Float64,
                            rng:: AbstractRNG):: Tuple{Vector{Nucleation}, Float64}
    n = rand(rng, Poisson(mean_nucleations))
    new_sites = sample(rng, n, space) |> fv_filter(existing_bubbles)
    fv_ratio = length(new_sites) / n
    nucleation_times = rand(rng, Uniform(t0, t0 + Δt), length(new_sites))
    nucleations = [Nucleation((time=t, site=p)) for (t, p) in zip(nucleation_times, new_sites)]
    sort!(nucleations)
    return nucleations, fv_ratio
end

abstract type NucleationLaw end

Base.start(nl:: NucleationLaw) = throw("Not Implemented for $(typeof(nl))")
Base.done(nl:: NucleationLaw) = throw("Not Implemented for $(typeof(nl))")
Base.next(nl:: NucleationLaw) = throw("Not Implemented for $(typeof(nl))")

function evolve(nucleation_law:: NucleationLaw, space:: AbstractSpace, 
                initial_state:: BubblesSnapshot = BubblesSnapShot(), 
                rng:: Union{AbstractRNG, Nothing} = nothing,
                termination_strategy:: Function = (_, _) -> false)
    if rng ≡ nothing
        rng = Random.default_rng()
    end
    state = initial_state
    for (Δt, λ) in nucleation_law
        bubbles = current_bubbles(state)
        new_nucs = sample_nucleations(Δt, λ, space, bubbles, state.t, rng)
        state = evolve(state, new_nucs, Δt)
        if termination_strategy(state, space)
            break
        end
    end                    
end

end

end