

function all_limits!(limits_buffer:: Vector{Tuple{Float64, Float64}}, intervals:: Vector{PeriodicInterval}):: AbstractVector{Tuple{Float64, Float64}}
    for (i, interval) in enumerate(intervals)
        limits_buffer[2i - 1] = (1., interval.ϕ1)
        limits_buffer[2i] = (-1., mod2π(interval.ϕ1 + interval.Δ))
    end
    return @views sort!(limits_buffer[1: 2 * length(intervals)], by=z -> z[2])
end

function periodic_intersection!(intervals:: Vector{PeriodicInterval}, 
                                limits_buffer:: Vector{Tuple{Float64, Float64}}, 
                                intersection_buffer:: Vector{PeriodicInterval}):: AbstractVector{PeriodicInterval}
    f = sum((0. ∈ interval) for interval in intervals)
    N = length(intervals)
    lims = all_limits!(limits_buffer, intervals)
    i = 1 
    f == N && begin 
            intersection_buffer[i] = PeriodicInterval(lims[end][2], mod2π(lims[1][2] - lims[end][2]))
            i += 1
        end
    for ((d, ϕ1), (_, ϕ2)) in partition(lims, 2, 1)
        f += d
        f == N && begin 
            intersection_buffer[i] = PeriodicInterval(ϕ1, mod2π(ϕ2 - ϕ1))
            i += 1
        end
    end
    @views return intersection_buffer[1:i - 1]
end
