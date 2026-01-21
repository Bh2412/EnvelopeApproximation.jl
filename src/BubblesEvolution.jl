module BubblesEvolution

using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubbleBasics: Vec3, Point3, coordinates
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
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

speed_of_light_profile(t:: Float64, c:: Float64 = 1.):: Float64 = c * t 

BubblesSnapShot(nucleations:: Vector{Nucleation}, t:: Float64) = BubblesSnapShot(nucleations, t, speed_of_light_profile)
BubblesSnapShot() = BubblesSnapShot(Vector{Nucleation}(), 0., speed_of_light_profile)

function at_earlier_time(snap:: BubblesSnapShot, t:: Float64):: BubblesSnapShot
    nucleations = filter(nuc -> nuc[:time] <= t, snap.nucleations)
    return BubblesSnapShot(nucleations, t, snap.radial_profile)
end

function evolve(snap:: BubblesSnapShot, nucleations:: Vector{Nucleation}, Δt:: Float64)
    return BubblesSnapShot([snap.nucleations..., nucleations...], snap.t + Δt, snap.radial_profile)
end

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

abstract type NucleationProcess end

function sample_nucleations(rng:: AbstractRNG, process:: NucleationProcess, space:: AbstractSpace, boundary_condition:: BoundaryCondition):: Vector{Nucleation}
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
                   space:: AbstractSpace, boundary_condition:: Periodic; padding:: Float64):: BubblesSnapShot
    nucleations = sample_nucleations(rng, nucleation_process, space, boundary_condition; padding=padding)
    return BubblesSnapShot(nucleations, completion_time(nucleation_process), radial_profile(nucleation_process))
end

end