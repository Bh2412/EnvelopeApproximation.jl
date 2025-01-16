module BubblesEvolution

using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubbleBasics: Vec3, Point3, coordinates
import Random.AbstractRNG
using StatsBase
import StatsBase.sample
import Distances.pairwise
import Base.∈
using Distributions
import Base.isless
import Random

Nucleation = @NamedTuple{time:: Float64, site:: Point3}
isless(n1:: Nucleation, n2:: Nucleation) = isless(n1[:time], n2[:time])

struct BubblesSnapShot
    nucleations:: Vector{Nucleation}
    t:: Float64
    radial_profile:: Function
end

export BubblesSnapShot

speed_of_light_profile(t:: Float64, c:: Float64 = 1.):: Float64 = c * t 

BubblesSnapShot(nucleations:: Vector{Nucleation}, t:: Float64) = BubblesSnapShot(nucleations, t, speed_of_light_profile)
BubblesSnapShot() = BubblesSnapShot(Vector{Nucleation}(), 0., speed_of_light_profile)

export BubblesSnapShot

function at_earlier_time(snap:: BubblesSnapShot, t:: Float64):: BubblesSnapShot
    nucleations = filter(nuc -> nuc[:time] <= t, snap.nucleations)
    return BubblesSnapShot(nucleations, t, snap.radial_profile)
end

export at_earlier_time

function evolve(snap:: BubblesSnapShot, nucleations:: Vector{Nucleation}, Δt:: Float64)
    return BubblesSnapShot([snap.nucleations..., nucleations...], snap.t + Δt, snap.radial_profile)
end

export evolve

function current_bubbles!(snap:: BubblesSnapShot, t:: Float64, 
                          bubbles_buffer:: Vector{Bubble}):: Bubbles
    i = 1
    for nuc in at_earlier_time(snap, t).nucleations
        bubbles_buffer[i] = Bubble(nuc[:site], snap.radial_profile(t - nuc[:time]))
        i += 1
    end
    return @views bubbles_buffer[1:(i-1)]
end

function current_bubbles!(snap:: BubblesSnapShot, 
                          bubbles_buffer:: Vector{Bubble}):: Bubbles
    t = snap.t
    return current_bubbles!(snap, t, bubbles_buffer)
end

function current_bubbles(snap:: BubblesSnapShot):: Bubbles
    t = snap.t
    buffer = Vector{Bubble}(undef, length(snap.nucleations))
    return current_bubbles!(snap, t, buffer)
end

function current_bubbles(snap:: BubblesSnapShot, t:: Float64):: Bubbles
    return current_bubbles(at_earlier_time(snap, t))
end

export current_bubbles, current_bubbles!

abstract type AbstractSpace end

function sample(rng:: AbstractRNG, n:: Int64, space:: AbstractSpace):: Vector{Point3} 
    buffer = Vector{Point3}(undef, n)
    return sample!(rng, n, space, buffer)
end

function sample!(rng:: AbstractRNG, n:: Int64, space:: AbstractSpace, points_buffer:: Vector{Point3}):: AbstractVector{Point3}
    throw("Cant sample from abstract space $space")
end

export sample!

struct BallSpace <: AbstractSpace
    radius:: Float64
    center:: Point3
end


const RADIAL_DISTRIBUTION:: Uniform{Float64} = Uniform(0., 1.)
const AZYMUTHAL_DISTRIBUTION:: Uniform{Float64} = Uniform(0., 2π)
const POLAR_DISTRIBUTION:: Uniform{Float64} = Uniform(-1., 1.)

export BallSpace

function sample(rng:: AbstractRNG, n:: Int64, space:: BallSpace):: Vector{Point3}
    # r^3 is distributed uniformly over (0, 1)
    r = rand(rng, Uniform(0., space.radius ^ 3), n) .^ (1 / 3)
    # ϕ is distributed uniformly over (0, 2π)
    ϕ = rand(rng, Uniform(0., 2π) , n)
    # μ is distributed uniformly over (-1., 1.) 
    μ = rand(rng, Uniform(-1., 1.), n)
    v = begin
        s = (x -> sqrt(1 - x^2)).(μ)
        @. Vec3(r * s * cos(ϕ), r * s * sin(ϕ), r * μ)
    end
    return @. (space.center, ) + v
end

function sample!(rng:: AbstractRNG, n:: Int64, space:: BallSpace, points_buffer:: Vector{Point3}):: AbstractVector{Point3}
    R = space.radius
    x0, y0, z0 = space.center.coordinates
    @inbounds for i in 1:n
        r = R * (rand(rng, RADIAL_DISTRIBUTION) ^ (1 // 3))
        ϕ = rand(rng, AZYMUTHAL_DISTRIBUTION)
        μ = rand(rng, POLAR_DISTRIBUTION)
        s = sqrt(1 - μ ^ 2)
        points_buffer[i] = Point3(Vec3((x0 + r * s * cos(ϕ), y0 + r * s * sin(ϕ), z0 + r * μ)))
    end
    return @views points_buffer[1:n]
end

function false_vacuum_filter!(sites:: Vector{Point3}, existing_bubbles:: Bubbles):: Vector{Point3}
    return filter!(s -> !any(s ∈ bubble for bubble in existing_bubbles), sites)
end

function sample_nucleations(Δt:: Float64,
                            mean_nucleations:: Float64, 
                            space:: AbstractSpace,
                            existing_bubbles:: Bubbles,
                            t0:: Float64,
                            rng:: AbstractRNG):: Tuple{Vector{Nucleation}, Float64}
    n = rand(rng, Poisson(mean_nucleations))
    @debug "A total of $n nucleations was sampled in accordance with the expected mean of $mean_nucleations"
    new_sites = false_vacuum_filter!(sample(rng, n, space), existing_bubbles)
    fv_ratio = length(new_sites) / n
    @debug "$(fv_ratio * 100)% of the sampled sites are within the true vacuum"
    nucleation_times = rand(rng, Uniform(t0, t0 + Δt), length(new_sites))
    nucleations = [Nucleation((time=t, site=p)) for (t, p) in zip(nucleation_times, new_sites)]
    sort!(nucleations)
    return nucleations, fv_ratio
end

abstract type NucleationLaw end

include("ExponentialGrowth.jl")

function bounding_bubbles(snapshot:: BubblesSnapShot, Δt:: Float64):: Bubbles
    t = snapshot.t
   return Bubbles([Bubble(nuc[:site], snapshot.radial_profile(t + Δt - nuc[:time])) for nuc in snapshot.nucleations]) 
end
                                                                                            
function evolve(nucleation_law:: NL, space:: S;
                initial_state:: BubblesSnapShot = BubblesSnapShot(), 
                rng:: Union{AbstractRNG, Nothing} = nothing,
                termination_strategy:: Function = (_, _, _) -> false) where {NL <: NucleationLaw, S <: AbstractSpace}
    if rng ≡ nothing
        rng = Random.default_rng()
    end
    state = initial_state
    @info "Initiating PT of $state"
    for (Δt, λ) in nucleation_law
        bubbles = bounding_bubbles(state, Δt)
        new_nucs, fv_ratio = sample_nucleations(Δt, λ, space, bubbles, state.t, rng)
        state = evolve(state, new_nucs, Δt)
        if termination_strategy(state, space, fv_ratio)
            break
        end
    end 
    return state                   
end

export evolve

end