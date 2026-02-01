export PeriodicInterval

"""
   PeriodicInterval

A structure representing an interval on a circle (a periodic domain).

A `PeriodicInterval` describes an arc on a circle, defined by a starting angle `ϕ1`
and an angular span `Δ` in the counterclockwise direction.

# Fields
- `ϕ1::Float64`: The starting angle in radians
- `Δ::Float64`: The angular span in radians (always non-negative)

# See Also
- [`∈`](@ref): Function to test if an angle is within a periodic interval.
- [`complement`](@ref): Function to create the complement of a periodic interval.
- [`a`](@ref): Function to get the starting angle of a periodic interval.
- [`b`](@ref): Function to get the ending angle of a periodic interval.
"""
struct PeriodicInterval
    ϕ1:: Float64
    Δ:: Float64
end

"""
   mod2π(ϕ::Float64)

Maps an angle to the range [0, 2π).

# Arguments
- `ϕ::Float64`: The angle in radians

# Returns
The equivalent angle in the range [0, 2π).
"""
mod2π(ϕ:: Float64) = mod(ϕ, 2π)

"""
   ∈(ϕ::Float64, p::PeriodicInterval)::Bool

Determines whether an angle is contained within a periodic interval.

# Arguments
- `ϕ::Float64`: The angle to test, in radians
- `p::PeriodicInterval`: The periodic interval

# Returns
`true` if the angle is within the interval, `false` otherwise.

# Notes
This function handles the periodic nature of angles correctly.
"""
∈(ϕ:: Float64, p:: PeriodicInterval):: Bool = mod2π(ϕ - p.ϕ1) <= p.Δ

"""
   approxempty(p::PeriodicInterval)::Bool

Tests if a periodic interval is approximately empty (zero span).

# Arguments
- `p::PeriodicInterval`: The periodic interval to test

# Returns
`true` if the interval has a span approximately equal to 0, `false` otherwise.
"""
approxempty(p:: PeriodicInterval):: Bool = p.Δ ≈ 0.

"""
   approxentire(p::PeriodicInterval)::Bool

Tests if a periodic interval approximately covers the entire circle.

# Arguments
- `p::PeriodicInterval`: The periodic interval to test

# Returns
`true` if the interval has a span approximately equal to 2π, `false` otherwise.
"""
approxentire(p:: PeriodicInterval):: Bool = p.Δ ≈ 2π 

"""
   EmptyArc

A constant representing an empty periodic interval (zero span).
"""
const EmptyArc:: PeriodicInterval = PeriodicInterval(0., 0.)

"""
   FullCircle

A constant representing a periodic interval that covers the entire circle.
"""
const FullCircle:: PeriodicInterval = PeriodicInterval(0., 2π)

"""
   complement(p::PeriodicInterval)::PeriodicInterval

Computes the complement of a periodic interval.

# Arguments
- `p::PeriodicInterval`: The periodic interval

# Returns
A new `PeriodicInterval` representing the complement of the input interval.

# Notes
- The complement of an empty interval is the full circle.
- The complement of the full circle is an empty interval.
- For other intervals, the complement starts at the end of the original interval
 and spans the remainder of the circle.
"""
function complement(p:: PeriodicInterval):: PeriodicInterval
    if approxempty(p)
        return FullCircle
    elseif approxentire(p)
        return EmptyArc
    else 
        return PeriodicInterval(mod2π(p.ϕ1 + p.Δ), 2π - p.Δ)
    end
end

"""
   a(p::PeriodicInterval)

Gets the starting angle of a periodic interval.

# Arguments
- `p::PeriodicInterval`: The periodic interval

# Returns
The starting angle `ϕ1` in radians.
"""
function a(p:: PeriodicInterval)
    return p.ϕ1
end

"""
   b(p::PeriodicInterval)

Gets the ending angle of a periodic interval.

# Arguments
- `p::PeriodicInterval`: The periodic interval

# Returns
The ending angle in radians, mapped to the range [0, 2π).
"""
function b(p:: PeriodicInterval)
    return mod2π(p.ϕ1 + p.Δ)
end
