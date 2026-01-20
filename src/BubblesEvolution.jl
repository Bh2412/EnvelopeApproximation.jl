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
using LinearAlgebra

export Nucleation, BubblesSnapShot, sample_PT, current_bubbles, at_earlier_time, ExponentialGrowthProcess, 
    BallSpace, volume, appropriate_t_end

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

export volume

function volume(space:: AbstractSpace):: Float64
    throw("Cant compute volume of abstract space $space")
end

function ∈(p:: Point3, space:: AbstractSpace):: Bool
    throw("Cant check membership of point $p in abstract space $space")
end

struct BoxSpace <: AbstractSpace
    L::Float64        # Side length of the cube
    center::Point3    
    
    BoxSpace(L::Float64, center::Point3 = Point3(0., 0., 0.)) = new(L, center)
end

volume(s::BoxSpace) = s.L^3

∈(p:: Point3, box_space:: BoxSpace) = begin
    d = p - box_space.center
    half_L = box_space.L / 2
    dx = abs(coordinates(d)[1])
    dy = abs(coordinates(d)[2])
    dz = abs(coordinates(d)[3])
    return (dx <= half_L) && (dy <= half_L) && (dz <= half_L)
end

function sample(rng::AbstractRNG, n::Int64, space::BoxSpace)::Vector{Point3}
    points = Vector{Point3}(undef, n)
    
    L = space.L
    center = space.center
    
    for i in 1:n
        dx = (rand(rng) - 0.5) * L
        dy = (rand(rng) - 0.5) * L
        dz = (rand(rng) - 0.5) * L
        
        points[i] = center + Vec3(dx, dy, dz)
    end
    
    return points
end     

"""
    bounding_box(space::AbstractSpace)::BoxSpace
Returns the smallest BoxSpace (with PBC capabilities) that contains the given space.
"""
function bounding_box(space::AbstractSpace)::BoxSpace
    throw("bounding_box not implemented for $(typeof(space))")
end

struct BallSpace <: AbstractSpace
    radius:: Float64
    center:: Point3
end

∈(p:: Point3, ball_space:: BallSpace) = norm(p - ball_space.center) <= ball_space.radius
volume(space:: BallSpace):: Float64 = (4 / 3) * π * space.radius ^ 3

function bounding_box(space:: BallSpace):: BoxSpace
    r = space.radius
    c = space.center
    return BoxSpace(2 * r, c)
end

const RADIAL_DISTRIBUTION:: Uniform{Float64} = Uniform(0., 1.)
const AZYMUTHAL_DISTRIBUTION:: Uniform{Float64} = Uniform(0., 2π)
const POLAR_DISTRIBUTION:: Uniform{Float64} = Uniform(-1., 1.)

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

abstract type NucleationProcess end

function sample_nucleations(rng:: AbstractRNG, process:: NucleationProcess, space:: AbstractSpace):: Vector{Nucleation}
    throw(ErrorException("sample_nucleations not implemented for $(typeof(process))"))
end

function radial_profile(process:: NucleationProcess):: Function
    throw(ErrorException("radial_profile not implemented for $(typeof(process))"))
end

function completion_time(process:: NucleationProcess):: Float64
    throw(ErrorException("completion_time not implemented for $(typeof(process))"))
end

include("ExponentialGrowth.jl")
                                                                                           
function sample_PT(rng:: AbstractRNG, 
                   nucleation_process:: NucleationProcess, 
                   space:: AbstractSpace; padding:: Float64):: BubblesSnapShot
    nucleations = sample_nucleations(rng, nucleation_process, space; padding=padding)
    return BubblesSnapShot(nucleations, completion_time(nucleation_process), radial_profile(nucleation_process))
end

export sample_PT

end