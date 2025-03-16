const RTOL:: Float64 = 1e-12

"""
    intersecting(bubble1::Bubble, bubble2::Bubble)::Bool

Determines whether two bubbles have an intersecting volume.

This function checks if two bubbles intersect by comparing the Euclidean distance
between their centers with the sum of their radii.

# Arguments
- `bubble1::Bubble`: The first bubble
- `bubble2::Bubble`: The second bubble

# Returns
`true` if the bubbles have a non-trivial intersection (overlapping volume),
`false` if they are separate or just tangent to each other.

# Notes
Bubbles that are exactly tangent (where distance equals the sum of radii within
a relative tolerance `RTOL`) are considered non-intersecting. This avoids treating
bubbles that just touch at a point as having an intersection.

# See Also
- [`Bubble`](@ref): The representation of a bubble with a center and radius.
- [`euc`](@ref): Function to calculate Euclidean distance between points.
"""
function intersecting(bubble1:: Bubble, bubble2:: Bubble):: Bool
    d = euc(bubble1.center, bubble2.center)
    radius_sum = bubble1.radius + bubble2.radius
    isapprox(d, radius_sum; rtol=RTOL) && return false
    return d < radius_sum
end

"""
    IntersectionDome

A structure representing the intersection geometry between bubbles.

An `IntersectionDome` describes a dome-shaped region defined by a height `h`, 
a unit normal vector `n̂`, and a boolean flag `dome_like` indicating its orientation.

# Fields
- `h::Float64`: This distance between the center of a bubble to the intersection surface
- `n̂::Vec3`: The unit normal vector pointing in the direction of the dome's axis
- `dome_like::Bool`: Orientation flag; `true` indicates a dome that is a spherical sector, 
    `false` indicates that the complement of the dome (within the bubble) is dome-like.

# See Also
- [`complement`](@ref): Function to create the complement of a dome.
- [`polar_intersection_region`](@ref): Function to calculate intersection regions for a dome.
"""
struct IntersectionDome
    h:: Float64
    n̂:: Vec3
    dome_like:: Bool
end

"""
    IntersectionDome(n::Vec3, dome_like::Bool)

Construct an `IntersectionDome` from a non-normalized direction vector and orientation flag.

This constructor normalizes the direction vector, computes the height as the vector's magnitude,
and creates an `IntersectionDome` with the specified orientation.

# Arguments
- `n::Vec3`: The non-normalized direction vector
- `dome_like::Bool`: Orientation flag

# Returns
A new `IntersectionDome` with height equal to the magnitude of `n`, 
and direction set to the normalized version of `n`.
"""
function IntersectionDome(n:: Vec3, dome_like:: Bool)
    h = norm(n)
    return IntersectionDome(h, n / h, dome_like)
end

"""
    complement(dome::IntersectionDome)

Creates the complement of an intersection dome.

Returns a new `IntersectionDome` with the same height and direction but with
the opposite orientation (inverting the `dome_like` flag).

# Arguments
- `dome::IntersectionDome`: The original intersection dome

# Returns
A new `IntersectionDome` representing the complement of the input dome.
"""
function complement(dome:: IntersectionDome)
    return IntersectionDome(dome.h, dome.n̂, ~dome.dome_like)
end

"""
    *(rotation::SMatrix{3, 3, Float64}, arc::IntersectionDome)::IntersectionDome

Applies a rotation matrix to an intersection dome.

This operator rotates the direction of the dome while preserving its height
and orientation flag.

# Arguments
- `rotation::SMatrix{3, 3, Float64}`: A 3×3 rotation matrix
- `arc::IntersectionDome`: The dome to be rotated

# Returns
A new `IntersectionDome` with the same height and orientation flag but with
the direction vector rotated according to the given rotation matrix.
"""
function *(rotation:: SMatrix{3, 3, Float64}, arc:: IntersectionDome):: IntersectionDome
    return IntersectionDome(arc.h, rotation * arc.n̂, arc.dome_like)
end

export IntersectionDome

"""
    λ(r1::Float64, r2::Float64, d::Float64)::Float64

Calculates the distance from the center of sphere 1 to the intersection plane between two spheres.

This function computes the parameter λ which has magnitude equal to the distance from the center of 
the first sphere to the plane of intersection between two spheres. Its sign determines whether
that plane is between the two sphere's centers or not

The formula is derived from the geometry of sphere-sphere intersections:
    λ = (d² + r₁² - r₂²) / (2d)

where:
- d is the distance between sphere centers
- r₁ is the radius of the first sphere
- r₂ is the radius of the second sphere

# Arguments
- `r1::Float64`: The radius of the first sphere
- `r2::Float64`: The radius of the second sphere
- `d::Float64`: The distance between the centers of the two spheres

# Returns
The distance from the center of sphere 1 to the intersection plane.

# Notes
This parameter is used to determine the geometry of the intersection between spheres.
When λ > 0, the intersection forms a dome-like shape from the perspective of sphere 1.
This means this function is assymetric in r₁ and r₂.

# References
Weisstein, Eric W. "Sphere-Sphere Intersection." From MathWorld--A Wolfram Web Resource.
https://mathworld.wolfram.com/Sphere-SphereIntersection.html
"""
function λ(r1:: Float64, r2:: Float64, d:: Float64) 
    return (d^2 + r1^2 - r2^2) / 2d
end

"""
    ∩(bubble1::Bubble, bubble2::Bubble)::Tuple{IntersectionDome, IntersectionDome}

Computes the intersection geometry between two bubbles.

Given two bubbles, this function calculates the pair of intersection domes that
represent how each bubble intersects with the other.

# Arguments
- `bubble1::Bubble`: The first bubble
- `bubble2::Bubble`: The second bubble

# Returns
A tuple containing two `IntersectionDome` objects:
- The first element represents how `bubble2` intersects with `bubble1`
- The second element represents how `bubble1` intersects with `bubble2`

# Notes
This function assumes neither bubble is completely contained within the other.
The dome-like property of each intersection dome is determined by the `λ` function,
which calculates a parameter related to the relative positions and sizes of the bubbles.

# See Also
- [`Bubble`](@ref): The representation of a bubble with a center and radius.
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`λ`](@ref): Function used to compute intersection parameters.
"""
function ∩(bubble1:: Bubble, bubble2:: Bubble):: Tuple{IntersectionDome, IntersectionDome}
    # This function assumes neither of the bubbles is contained in the other
    n = bubble2.center - bubble1.center
    d = norm(n)
    n̂ = n / d 
    _λ = λ(bubble1.radius, bubble2.radius, d)
    n1 = _λ * n̂
    n2 = -n + n1
    domelike1 = _λ > 0.
    domelike2 = λ(bubble2.radius, bubble1.radius, d) > 0.
    return (IntersectionDome(n1, domelike1), IntersectionDome(n2, domelike2))
end

"""
    intersection_domes(bubbles::Bubbles)::Dict{Int, Vector{IntersectionDome}}

Computes all pairwise intersections between a collection of bubbles.

This function calculates how each bubble in the collection intersects with every other bubble,
and organizes the results in a dictionary mapping bubble indices to their intersection domes.

# Arguments
- `bubbles::Bubbles`: A collection of bubbles

# Returns
A dictionary where:
- Keys are indices of bubbles in the collection
- Values are vectors of `IntersectionDome` objects representing how other bubbles
  intersect with the bubble at the given index

# Notes
For each pair of intersecting bubbles, two intersection domes are created:
one representing how the second bubble intersects with the first,
and another representing how the first bubble intersects with the second.

# See Also
- [`Bubbles`](@ref): The collection type for bubbles.
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`intersecting`](@ref): Function that determines if two bubbles intersect.
- [`∩`](@ref): Function that calculates the intersection between two bubbles.
"""
function intersection_domes(bubbles:: Bubbles):: Dict{Int, Vector{IntersectionDome}}
    d = Dict{Int, Vector{IntersectionDome}}()
    for i in eachindex(bubbles)
        d[i] = Vector{IntersectionDome}()
    end
    for (i, bubble1) in enumerate(bubbles)
        for (j̃, bubble2) in enumerate(bubbles[(i + 1):end])
            j = j̃ + i
            if intersecting(bubble1, bubble2)
                intersection1, intersection2 = bubble1 ∩ bubble2
                for (k, intersection) in ((i, intersection1), (j, intersection2))
                    push!(d[k], intersection)
                end
            else
                continue
            end        
        end
    end
    return d
end

"""
    ⊆(bubble::Bubble, ball_space::BallSpace)::Bool

Determines whether a bubble is contained entirely within a ball space.

This function checks if a bubble is completely inside a ball space by comparing
the distance between their centers with the difference of their radii.

# Arguments
- `bubble::Bubble`: The bubble to check
- `ball_space::BallSpace`: The containing ball space

# Returns
`true` if the bubble is entirely contained within the ball space,
`false` otherwise.

# Notes
A bubble exactly tangent to the inner surface of the ball space (where distance equals
radius difference within a relative tolerance `RTOL`) is considered contained.

# See Also
- [`Bubble`](@ref): The representation of a bubble with a center and radius.
- [`BallSpace`](@ref): The representation of the containing ball space.
- [`euc`](@ref): Function to calculate Euclidean distance between points.
"""
function ⊆(bubble:: Bubble, ball_space:: BallSpace):: Bool
    d = euc(bubble.center, ball_space.center)
    (isapprox(d, ball_space.radius - bubble.radius; rtol=RTOL)) && return true
    return euc(bubble.center, ball_space.center) < abs(ball_space.radius - bubble.radius)
end

"""
    ∩(bubble::Bubble, ball_space::BallSpace)::IntersectionDome

Computes the intersection geometry between a bubble and a ball space.

This function calculates the intersection dome that represents how a bubble
intersects with the containing ball space.

# Arguments
- `bubble::Bubble`: The bubble
- `ball_space::BallSpace`: The containing ball space

# Returns
An `IntersectionDome` object representing the intersection between the bubble and ball space.

# Notes
This function provides the mathematical intersection of the bubble and ball space.
The intersection complement is guaranteed to be dome-like, so the returned dome has
`dome_like` set to `false`.

# See Also
- [`Bubble`](@ref): The representation of a bubble with a center and radius.
- [`BallSpace`](@ref): The representation of the containing ball space.
- [`IntersectionDome`](@ref): The representation of intersections.
- [`λ`](@ref): Function used to compute intersection parameters.
"""
function ∩(bubble:: Bubble, ball_space:: BallSpace):: IntersectionDome # This gives the *intersection*, as it is mathematically defined, of the bubble and a ball_space
    n = bubble.center - ball_space.center
    d = norm(n)
    n̂ = n / d 
    h = -λ(bubble.radius, ball_space.radius, d)  # The intersection *complement* is guarenteed to be dome like.
    return IntersectionDome(h, n̂, false)
end

"""
    intersection_domes(bubbles::Bubbles, ball_space::BallSpace)::Dict{Int, Vector{IntersectionDome}}

Computes all intersections between a collection of bubbles and a containing ball space.

This function extends `intersection_domes` to also account for the intersections between
each bubble and the containing ball space boundary.

# Arguments
- `bubbles::Bubbles`: A collection of bubbles
- `ball_space::BallSpace`: The containing ball space

# Returns
A dictionary where:
- Keys are indices of bubbles in the collection
- Values are vectors of `IntersectionDome` objects representing both:
  1. How other bubbles intersect with the bubble at the given index
  2. How the ball space boundary intersects with the bubble (if applicable)

# Notes
For bubbles not fully contained within the ball space, an additional intersection
dome is added representing the complement of the intersection with the ball space.
This is equivalent to having a reflective bubble that reaches the ball space surface.

# See Also
- [`Bubbles`](@ref): The collection type for bubbles.
- [`BallSpace`](@ref): The representation of the containing ball space.
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`⊆`](@ref): Function that determines if a bubble is contained in a ball space.
- [`∩`](@ref): Functions that calculate intersections.
- [`complement`](@ref): Function to create the complement of a dome.
"""
function intersection_domes(bubbles:: Bubbles, ball_space:: BallSpace)
    domes = intersection_domes(bubbles)
    for (i, bubble) in enumerate(bubbles)
        if ~(bubble ⊆ ball_space)
            # Since in all other cases an "intersection_dome" is a region excluded from integration,  
            # and because we want to integrate the region of the bubble within the ball_space, we take here the complement of the interection.
            # This is equivalent to having a reflective bubble that collides with a bubble that reaches the surface.
            push!(domes[i], complement(bubble ∩ ball_space))  
        end
    end
    return domes
end

export intersection_domes
