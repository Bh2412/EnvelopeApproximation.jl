module EnvelopeApproximation

import Meshes.Point3

export Point3

module BubbleBasics

using Distances
import Meshes: Point3, coordinates
import Base.length
import Base.∈

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

Base.getindex(b:: Bubbles, index:: Int64):: Bubble = b.bubbles[index]

euclidean = Euclidean()
euc(point1:: Point3, point2:: Point3):: Float64 = euclidean(coordinates.([point1, point2])...)

export euc

∈(point:: Point3, bubble:: Bubble) :: Bool = euc(point, bubble.center) <= bubble.radius
∈(point:: Point3, bubbles:: Bubbles):: Bool = any(point .∈ bubbles.bubbles)

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

include("BubblesIntegration.jl")

include("StressEnergyTensor.jl")

include("GravitationalPotentials.jl")

include("Visualization.jl")

end
