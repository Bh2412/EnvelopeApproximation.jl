using .RayCastingEnvelopeIntegration: EnvelopeSource

"""
    TrueVacuumVolume(times)

Kernel that computes the true-vacuum volume at the requested sorted `times`.

For a marker with angular weight `wΩ`, the accumulated volume at time `t` is
`wΩ * τ³ / 3`, where `τ = min(t - tₙ, τ_stop)`. The contribution is zero
before the source nucleates and remains constant after the ray stops.
"""
struct TrueVacuumVolume{Times<:AbstractVector{<:Real}} <: Kernel
    times::Times

    function TrueVacuumVolume(times::Times) where {Times<:AbstractVector{<:Real}}
        issorted(times) ||
            throw(ArgumentError("times must be sorted in nondecreasing order"))
        return new{Times}(times)
    end
end

function allocate_accumulant(kernel::TrueVacuumVolume)
    return zeros(Float64, length(kernel.times))
end

function accumulate_ray!(
    volumes::Vector{Float64},
    kernel::TrueVacuumVolume,
    source::EnvelopeSource,
    direction,
    wΩ::Float64,
    τ_stop::Float64,
)
    first_time = searchsortedlast(kernel.times, source.time) + 1

    @inbounds for a in first_time:lastindex(kernel.times)
        τ = min(Float64(kernel.times[a] - source.time), τ_stop)
        volumes[a] += wΩ * τ^3 / 3.0
    end

    return nothing
end

export TrueVacuumVolume
