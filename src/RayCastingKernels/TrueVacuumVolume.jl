using .RayCastingEnvelopeIntegration: EnvelopeSource

"""
    TrueVacuumVolume(times; v=1.0)

Kernel that computes the true-vacuum volume at the requested sorted `times`.

For a marker with angular weight `wΩ`, the accumulated volume at time `t` is
`wΩ * (vτ)³ / 3`, where `τ = min(t - tₙ, τ_stop)`. The contribution is zero
before the source nucleates and remains constant after the ray stops.
"""
struct TrueVacuumVolume{Times<:AbstractVector{<:Real}} <: Kernel
    times::Times
    v::Float64

    function TrueVacuumVolume(
        times::Times;
        v::Real=1.0,
    ) where {Times<:AbstractVector{<:Real}}
        issorted(times) ||
            throw(ArgumentError("times must be sorted in nondecreasing order"))
        v > 0 || throw(ArgumentError("v must be positive"))
        return new{Times}(times, Float64(v))
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
        volumes[a] += wΩ * (kernel.v * τ)^3 / 3.0
    end

    return nothing
end

export TrueVacuumVolume
