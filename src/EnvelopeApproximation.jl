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
Point3(x:: Float64, y:: Float64, z:: Float64) = Point3(Vec3((x, y, z)))

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

Bubbles = AbstractVector{Bubble}

export Bubbles

euc(p1:: Point3, p2:: Point3):: Float64 = norm(coordinates(p1) - coordinates(p2))

export euc

function ≲(a:: Float64, b:: Float64):: Bool
    return (a <= b) | (isapprox(a, b, atol=1e-12))
end

export ≲

∈(point:: Point3, bubble:: Bubble) :: Bool = euc(point, bubble.center) ≲ bubble.radius
∈(point:: Point3, bubbles:: Bubbles):: Bool = any(point in b for b in bubbles)

export ∈

function radii(bubbles:: Bubbles):: Vector{Float64}
    return [bubble.radius for bubble in bubbles]
end

export radii

function centers(bubbles:: Bubbles):: Vector{Point3}
    return [bubble.center for bubble in bubbles]
end

export centers

end

include("BubblesEvolution.jl")

include("FractionalFFT.jl")

include("FilonQuadrature.jl")

include("ChebyshevCFT/ChebyshevCFT.jl")

include("GeometricStressEnergyTensor.jl")

include("ChebyshevCFT/ChebyshevCFTDiagnostics.jl")

include("GravitationalPotentials.jl")

include("ISWPowerSpectrum.jl")

include("GravitationalWaves.jl")

include("Visualization.jl")

end
