

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
