

const lib_path = "$(dirname(pathof(EnvelopeApproximation)))/EnvelopeAnalysis/ring_dome_complement_intersection.so"

struct Vector3
    x:: Float64
    y:: Float64
    z:: Float64
end

function Vector3(v:: Vec3):: Vector3
    return Vector3(v[1], v[2], v[3])
end

"""
   ring_dome_complement_intersection(μ::Float64, R::Float64, n̂::Vector3, h::Float64, dome_like::Bool)::PeriodicInterval

Low-level FFI function that calls the Rust implementation to compute the intersection 
between an integration ring and the complement of a dome-like region.

# Arguments
- `μ::Float64`: The polar angle cosine defining the integration ring
- `R::Float64`: The radius of the bubble
- `n̂::Vector3`: The unit normal vector of the dome
- `h::Float64`: The height of the dome
- `dome_like::Bool`: Flag indicating the orientation of the dome

# Returns
A `PeriodicInterval` representing the azimuthal range of the intersection.

# Notes
This is a direct ccall to the Rust implementation for performance-critical calculations.
"""
ring_dome_complement_intersection(μ:: Float64, R:: Float64, n̂:: Vector3, h:: Float64, dome_like:: Bool):: PeriodicInterval = @ccall lib_path.ring_dome_intersect(μ:: Float64, R:: Float64, n̂:: Vector3, h:: Float64, dome_like:: Bool):: PeriodicInterval

"""
   ring_dome_complement_intersection(μ′::Float64, R::Float64, n̂′::Vec3, h::Float64, dome_like::Bool)::PeriodicInterval

Computes the intersection between an integration ring and the complement of a dome-like region.

This version handles the internal `Vec3` type, converting it to the FFI-compatible `Vector3` type.

# Arguments
- `μ′::Float64`: The polar angle cosine defining the integration ring in the rotated coordinate system
- `R::Float64`: The radius of the bubble
- `n̂′::Vec3`: The unit normal vector of the dome in the rotated coordinate system
- `h::Float64`: The height of the dome
- `dome_like::Bool`: Flag indicating the orientation of the dome

# Returns
A `PeriodicInterval` representing the azimuthal range of the intersection.

# Notes
The prime notation (μ′, n̂′) indicates that these values are in a rotated coordinate system.
"""
function ring_dome_complement_intersection(μ′:: Float64, R:: Float64, n̂′:: Vec3, h:: Float64, dome_like:: Bool):: PeriodicInterval
    return ring_dome_complement_intersection(μ′, R, Vector3(n̂′), h, dome_like)
end

"""
   ring_dome_complement_intersection(μ′::Float64, R::Float64, intersection′::IntersectionDome)::PeriodicInterval

Computes the intersection between an integration ring and the complement of a dome-like region.

This version takes an `IntersectionDome` object instead of separate parameters.

# Arguments
- `μ′::Float64`: The polar angle cosine defining the integration ring in the rotated coordinate system
- `R::Float64`: The radius of the bubble
- `intersection′::IntersectionDome`: The intersection dome in the rotated coordinate system

# Returns
A `PeriodicInterval` representing the azimuthal range of the intersection.

# Notes
The prime notation (μ′, intersection′) indicates that these values are in a rotated coordinate system.

# See Also
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`PeriodicInterval`](@ref): The representation of angular intervals on a circle.
"""
function ring_dome_complement_intersection(μ′:: Float64, R:: Float64, 
                                           intersection′:: IntersectionDome):: PeriodicInterval
    n̂′, h, dome_like = intersection′.n̂, intersection′.h, intersection′.dome_like
    return ring_dome_complement_intersection(μ′, R, n̂′, h, dome_like) 
end

function all_limits!(intervals:: AbstractVector{PeriodicInterval}, 
                     limits_buffer:: Vector{Tuple{Float64, Float64}}):: AbstractVector{Tuple{Float64, Float64}}
    i = 1
    for interval in intervals
        limits_buffer[2i - 1] = (1., interval.ϕ1)
        limits_buffer[2i] = (-1., mod2π(interval.ϕ1 + interval.Δ))
        i += 1
    end
    return @views sort!(limits_buffer[1: 2 * (i - 1)], by=z -> z[2])
end

"""
    periodic_intersection!(intervals:: AbstractVector{PeriodicInterval}, 
                           limits_buffer:: Vector{Tuple{Float64, Float64}}, 
                           intersection_buffer:: Vector{PeriodicInterval}):: AbstractVector{PeriodicInterval}

Calculate the intersection of a collection of periodic intervals on the circle [0, 2π).

This function assumes at least two intervals are input, and that no interval is of length >= 2π.
It efficiently computes the intersection by tracking interval boundaries and maintaining a count
of active intervals at each boundary crossing.

# Arguments
- `intervals`: A vector of `PeriodicInterval`s to intersect
- `limits_buffer`: Pre-allocated buffer for storing interval limits
- `intersection_buffer`: Pre-allocated buffer for storing the resulting intersection intervals

# Returns
- A slice of the `intersection_buffer` containing the intersection intervals

# Note
The function modifies the input buffers and returns a view into `intersection_buffer`.
In addition, this function assumes that the buffers are large enough to store the results.
The function does not check the buffer sizes and may cause undefined behavior if the buffers are too small.
In practice, this is guranteed by choosing buffers with at least twice the length of the input intervals.

# See Also
- [`PeriodicInterval`](@ref): A data structure representing an interval on the periodic domain [0, 2π)
"""
function periodic_intersection!(intervals:: AbstractVector{PeriodicInterval}, 
                                limits_buffer:: Vector{Tuple{Float64, Float64}}, 
                                intersection_buffer:: Vector{PeriodicInterval}):: AbstractVector{PeriodicInterval}
    lims = all_limits!(intervals, limits_buffer)
    # The region between the first interval edge and last interval edge
    x0 = mod2π((lims[1][2] + lims[end][2]) / 2 + π)
    f = sum((x0 ∈ interval) for interval in intervals)
    N = length(intervals)
    i = 1 
    f == N && begin 
            # Using that the length of the arc between the end point and start point is 2π - the opposit arc
            intersection_buffer[i] = PeriodicInterval(lims[end][2], 2π - (lims[end][2] - lims[1][2]))
            i += 1
        end
    for ((d, ϕ1), (_, ϕ2)) in partition(lims, 2, 1)
        f += d
        f == N && begin 
            # Using the fact that 0. < ϕ2 - ϕ1 < 2π due to sorting and the input assumptions 
            intersection_buffer[i] = PeriodicInterval(ϕ1, ϕ2 - ϕ1)
            i += 1
        end
    end
    @views return intersection_buffer[1:i - 1]
end

"""
    ring_domes_complement_intersection!(μ′::Float64, R::Float64, intersection_domes::Vector{IntersectionDome}, 
                                        arcs_buffer::Vector{PeriodicInterval}, 
                                        limits_buffer::Vector{Tuple{Float64, Float64}},
                                        intersection_buffer::Vector{PeriodicInterval})::AbstractVector{PeriodicInterval}

Calculate the intersection of the complements of multiple dome projections with a ring. 
The ring is parallel to the z direction, and is a portion of a bubble of radius `R`.

This function computes the angular regions of a ring with radius `R` that do not fall within 
any of the provided `intersection_domes`. The result represents the angular intervals on the ring 
that are outside all domes.

# Arguments
- `μ′::Float64`: The ring's parameter. I.E, the value `z/R - 1` for each of the points that make up a ring perpendicular to the z direction
    And is contained in the bubble of radius `R`.
- `R::Float64`: The radius of the bubble that contains the ring.
- `intersection_domes::Vector{IntersectionDome}`: Collection of domes. These domes are the portion of the intersection of bubbles that 
    lies within the input bubble of radius R
- `arcs_buffer::Vector{PeriodicInterval}`: Pre-allocated buffer for storing intermediate periodic intervals.
- `limits_buffer::Vector{Tuple{Float64, Float64}}`: Pre-allocated buffer for storing angle limits.
- `intersection_buffer::Vector{PeriodicInterval}`: Pre-allocated buffer for storing intersection results.

# Returns
A view into `intersection_buffer` containing the resulting periodic intervals representing 
the parts of the ring outside all domes.

# Notes
- If `intersection_domes` is empty, returns the full circle as there are no domes to exclude.
- If there's only one dome, directly returns the complement of its intersection with the ring.
- For multiple domes, computes the intersection of all complements efficiently using the provided buffers.
- Returns an empty view if the result is empty (the ring is completely covered by domes).

# See Also
- [`IntersectionDome`](@ref): The representation of intersections between bubbles.
- [`ring_dome_complement_intersection`](@ref): Function to find the complement of a single dome's intersection with a ring.
- [`periodic_intersection!`](@ref): Function used to compute the intersection of multiple periodic intervals.
"""
function ring_domes_complement_intersection!(μ′:: Float64, R:: Float64, intersection_domes:: Vector{IntersectionDome}, 
                                             arcs_buffer:: Vector{PeriodicInterval}, 
                                             limits_buffer:: Vector{Tuple{Float64, Float64}},
                                             intersection_buffer:: Vector{PeriodicInterval}):: AbstractVector{PeriodicInterval}
    isempty(intersection_domes) && begin 
        intersection_buffer[1] = FullCircle
        return @views intersection_buffer[1:1]
    end
    length(intersection_domes) == 1 && begin
        intersection_buffer[1] = ring_dome_complement_intersection(μ′, R, intersection_domes[1])
        return @views intersection_buffer[1:1]
    end
    i = 1
    for dome in intersection_domes
        p = ring_dome_complement_intersection(μ′, R, dome)
        approxempty(p) && return @views intersection_buffer[1:0]
        approxentire(p) && continue
        arcs_buffer[i] = p
        i += 1
    end
    i == 1 && ((intersection_buffer[1] = FullCircle); return @views intersection_buffer[1:1])
    i == 2 && return @views arcs_buffer[1:1]
    return @views periodic_intersection!(arcs_buffer[1:(i - 1)], limits_buffer, intersection_buffer)
end

function ring_domes_complement_buffers(n)
    arcs_buffer = PeriodicInterval[]
    limits_buffer = Tuple{Float64, Float64}[]
    intersection_buffer = Vector{PeriodicInterval}[]
    resize!(arcs_buffer, n)
    resize!(limits_buffer, 2n)
    resize!(intersection_buffer, n)
    return arcs_buffer, limits_buffer, intersection_buffer
end
