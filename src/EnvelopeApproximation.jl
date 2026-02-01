module EnvelopeApproximation

module BubbleBasics

using Distances
import Base: ∈, keys, +, -, lastindex, length, getindex
using StaticArrays
import LinearAlgebra: norm, ⋅
import Base: +, -

export Vec3, Point3, coordinates, with_origin, ⋅, +, -, Bubble, Bubbles, euc, ∈, radii, centers, ≲

Vec3 = SVector{3, Float64}

struct Point3
    coordinates:: SVector{3, Float64}
end

Point3(args...) = Point3(Vec3(args...))
Point3(x:: Float64, y:: Float64, z:: Float64) = Point3(Vec3((x, y, z)))

coordinates(p:: Point3):: SVector{3, Float64} = p.coordinates

+(p:: Point3, v:: Vec3):: Point3 = Point3(coordinates(p) .+ v)
-(p:: Point3, v:: Vec3):: Point3 = Point3(coordinates(p) .- v)
-(p1:: Point3, p2:: Point3):: Vec3 = coordinates(p1) .- coordinates(p2)

with_origin(p:: Point3, o:: Point3) = Point3(p - o)

struct Bubble
    center:: Point3
    radius:: Float64
end

Bubbles = AbstractVector{Bubble}

euc(p1:: Point3, p2:: Point3):: Float64 = norm(coordinates(p1) - coordinates(p2))

function ≲(a:: Float64, b:: Float64):: Bool
    return (a <= b) | (isapprox(a, b, atol=1e-12))
end

∈(point:: Point3, bubble:: Bubble) :: Bool = euc(point, bubble.center) ≲ bubble.radius
∈(point:: Point3, bubbles:: Bubbles):: Bool = any(point in b for b in bubbles)

function radii(bubbles:: Bubbles):: Vector{Float64}
    return [bubble.radius for bubble in bubbles]
end

function centers(bubbles:: Bubbles):: Vector{Point3}
    return [bubble.center for bubble in bubbles]
end

end

module Spaces

import EnvelopeApproximation.BubbleBasics: Vec3, Point3, coordinates
import Random.AbstractRNG
using StatsBase
import StatsBase.sample
import Base.∈
using Distributions
import Random
using LinearAlgebra

export AbstractSpace, BoxSpace, BallSpace, volume, sample, bounding_box, ∈

abstract type AbstractSpace end

function sample(rng:: AbstractRNG, n:: Int64, space:: AbstractSpace):: Vector{Point3} 
    buffer = Vector{Point3}(undef, n)
    return sample!(rng, n, space, buffer)
end

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
    dx, dy, dz = abs.(d)
    return (dx <= half_L) && (dy <= half_L) && (dz <= half_L)
end

function sample(rng:: AbstractRNG, space::BoxSpace):: Point3
    L = space.L
    c = coordinates(space.center)

    x = (rand(rng) - 0.5) * L + c[1]
    y = (rand(rng) - 0.5) * L + c[2]
    z = (rand(rng) - 0.5) * L + c[3]
    return Point3(x, y, z)
end

function sample(rng::AbstractRNG, n::Int64, space::BoxSpace)::Vector{Point3}
points = Vector{Point3}(undef, n)
    
    L = space.L
    c = coordinates(space.center)
    
    offset = 0.5 * L
    ll_x = c[1] - offset
    ll_y = c[2] - offset
    ll_z = c[3] - offset
    
    @inbounds for i in 1:n
        px = muladd(rand(rng), L, ll_x)
        py = muladd(rand(rng), L, ll_y)
        pz = muladd(rand(rng), L, ll_z)
        
        points[i] = Point3(px, py, pz)
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

function bounding_box(space::BoxSpace)::BoxSpace
    return space
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

function sample(rng:: AbstractRNG, space:: BallSpace):: Point3
    r = space.radius * cbrt(rand(rng))
    ϕ = rand(rng, Uniform(0, 2π))
    μ = rand(rng, Uniform(-1., 1.))
    sμ = √(1 - μ ^ 2)
    sϕ, cϕ = sincos(ϕ)
    return space.center + Vec3(r * cϕ * sμ, r * sϕ * sμ, r * μ)
end

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

end

module BoundaryConditions

export BoundaryCondition, Vacuum, Periodic

abstract type BoundaryCondition end

struct Vacuum <: BoundaryCondition end
struct Periodic <: BoundaryCondition end

end

include("BubblesEvolution.jl")

include("CFT/CFTInteface.jl")

include("CFT/ChebyshevCFT.jl")

include("CFT/QuadGKCFT.jl")

include("EnvelopeAnalysis.jl")

include("AngularIntegration/AngularIntegrationInterface.jl")

include("AngularIntegration/SphericalHarmonics.jl")

include("GravitationalWaves.jl")

include("Visualization.jl")

end
