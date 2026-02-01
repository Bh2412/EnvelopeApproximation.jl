"""
    polar_intersection_region(R::Float64, dome::IntersectionDome)::Tuple{Float64, Float64}

Calculates the polar angle boundaries of a dome's intersection region with a bubble.

Given a bubble of radius `R` and an intersection dome, this function computes
the range of cosines of polar angles where the dome intersects the bubble.

# Arguments
- `R::Float64`: The radius of the bubble
- `dome::IntersectionDome`: The intersection dome representing the intersection
  between bubbles

# Returns
A tuple `(min, max)` of Float64 values representing the cosines of the polar angles
that bound the intersection region.

# Notes
This function assumes that `dome.h < R` (the dome height is less than the radius),
which should be ensured before calling this function.

The calculation is performed by:
1. Computing the z-component of the dome's normal vector (`n̂_z`)
2. Computing the xy-component magnitude of the normal vector (`n̂_xy`)
3. Finding the central value of the cosine range using the ratio of dome height to radius
4. Calculating the span of the intersection region
And is based on geometrical analysis.

# See Also
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`polar_limits`](@ref): Function to collect all intersection region boundaries from multiple domes.
"""
function polar_intersection_region(R:: Float64, 
                                   dome:: IntersectionDome):: Tuple{Float64, Float64}
    # This function assumes h < R
    n̂_z = dome.n̂[3]
    n̂_xy = √(1 - n̂_z ^ 2)
    ratio = dome.h / R
    c = n̂_z * ratio
    Δ = n̂_xy * √(1 - ratio ^ 2)
    return (c - Δ, c + Δ)
end

"""
    polar_limits(R::Float64, domes::Vector{IntersectionDome})::Vector{Float64}

Calculates the cosines of polar angle limits of intersection regions between a bubble and multiple domes.

Given a bubble with radius `R` and a collection of intersection domes, this function
computes all distinct angular limits that define regions of intersection in polar coordinates.
The returned values represent the cosines of the polar angles that bound these regions.
A usual application of this function is to gather regions of polar angle in which any smooth integrand
over the bubble remains smooth.

# Arguments
- `R::Float64`: The radius of the bubble
- `domes::Vector{IntersectionDome}`: A collection of intersection domes representing the intersections
  between the bubble and other geometric structures

# Returns
A sorted vector of Float64 values in the range [-1, 1], representing the cosines of 
polar angles that define the boundaries of intersection regions. The vector always 
starts with -1 and ends with 1 to ensure the entire angular range is covered.

# See Also
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`polar_intersection_region`](@ref): Function that calculates intersection regions for a single dome.

# Notes
This function filters out values outside the valid range for cosines [-1, 1] and
ensures uniqueness of the boundary points.
"""
function polar_limits(R:: Float64, domes:: Vector{IntersectionDome}):: Vector{Float64}    
    regions = (x for dome in domes for t in polar_intersection_region(R, dome) for x in t if abs(x) <= 1.)
    regions = regions |> collect |> unique! |> sort!
    pushfirst!(regions, -1.)
    return push!(regions, 1.)
end