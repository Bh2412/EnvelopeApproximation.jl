module EnvelopeApproximation

import Meshes.Point3

export Point3

module BubbleBasics

import Meshes.Point3
import Base.length

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

include("GravitationalPotentials.jl")

end
