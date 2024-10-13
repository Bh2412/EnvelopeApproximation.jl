module EnvelopeApproximation

module BubbleBasics

using Distances
import Base: ∈, keys, +, -, lastindex, length, getindex
using StaticArrays
import LinearAlgebra: norm, ⋅
import Base: +, -

Vec3 = SVector{3, Float64}

export Vec3, ⋅

struct Point3
    coordinates:: SVector{3, Float64}
end

Point3(args...) = Point3(Vec3(args...))

export Point3

coordinates(p:: Point3):: SVector{3, Float64} = p.coordinates

export coordinates

+(p:: Point3, v:: Vec3):: Point3 = Point3(coordinates(p) .+ v)
-(p1:: Point3, v:: Vec3):: Point3 = Point3(coordinates(p) .- v)
-(p1:: Point3, p2:: Point3):: Vec3 = coordinates(p1) .- coordinates(p2)

export +, -

struct Bubble
    center:: Point3
    radius:: Float64
end

export Bubble

struct Bubbles
    bubbles:: Vector{Bubble}
end

function Bubbles(centers:: Vector{Point3}, 
                 radii:: Vector{Float64}):: Bubbles
    length(centers) != length(radii) && throw(ArgumentError("The bubble centers and bubble radii must have
                                                             the same length.")) 
    return Bubbles([Bubble(center, radius) for (center, radius) in zip(centers, radii)])
end

export Bubbles

function length(bubbles:: Bubbles):: Int64
    return length(bubbles.bubbles)
end

export length

function Base.iterate(bs:: Bubbles)
    return Base.iterate(bs.bubbles)
end

function Base.iterate(bs:: Bubbles, state)
    return Base.iterate(bs.bubbles, state)
end

lastindex(bs:: Bubbles) = lastindex(bs.bubbles)
keys(bs:: Bubbles) = keys(bs.bubbles)
getindex(bs:: Bubbles, i:: Int) = getindex(b.bubbles, i)

Base.getindex(b:: Bubbles, index:: Int64):: Bubble = b.bubbles[index]

euc(p1:: Point3, p2:: Point3):: Float64 = norm(coordinates(p1) - coordinates(p2))

export euc

function ≲(a:: Float64, b:: Float64):: Bool
    return (a <= b) | (isapprox(a, b, atol=1e-12))
end

export ≲

∈(point:: Point3, bubble:: Bubble) :: Bool = euc(point, bubble.center) ≲ bubble.radius
∈(point:: Point3, bubbles:: Bubbles):: Bool = any(point in b for b in bubbles.bubbles)

export ∈

function radii(bubbles:: Bubbles):: Vector{Float64}
    return [bubble.radius for bubble in bubbles.bubbles]
end

export radii

function centers(bubbles:: Bubbles):: Vector{Point3}
    return [bubble.center for bubble in bubbles.bubbles]
end

export centers

end

include("BubblesEvolution.jl")

include("SurfaceTesselation.jl")

include("StressEnergyTensor.jl")

include("StressEnergyTensorFFT.jl")

include("GeometricStressEnergyTensor.jl")

include("GravitationalPotentials.jl")

include("TemperatureFluctuations.jl")

include("Visualization.jl")

end
