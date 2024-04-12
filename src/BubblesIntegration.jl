module BubblesIntegration

import Meshes.Point3
import Base.length

export Point3

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

function radii(bubbles:: Bubbles):: Vector{Float64}
    return [bubble.radius for bubble in bubbles.bubbles]
end

export radii

function centers(bubbles:: Bubbles):: Vector{Point3}
    return [bubble.center for bubble in bubbles.bubbles]
end

export centers

include("SurfaceIntergration.jl")

export SurfaceIntegration

include("VolumeIntegration.jl")

export VolumeIntegration


end
