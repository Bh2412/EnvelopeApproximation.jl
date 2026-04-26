module BubbleBasics

using Distances
import Base: ∈, keys, +, -, *, lastindex, length, getindex
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
*(R:: AbstractMatrix{Float64}, p:: Point3):: Point3 = Point3(R * coordinates(p))

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