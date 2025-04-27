

const lib_path = "$(dirname(pathof(EnvelopeApproximation)))/GeometricStressEnergyTensor/ring_dome_complement_intersection.so"

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
