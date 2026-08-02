# Instantaneous Fourier stress tensor on a prescribed time grid.

"""
    TimeResolvedStressTensorAccumulant

Accumulant returned by [`ray_T_ij_at_times`](@ref). `A` has axes
`(tensor_component, wave_number, time)`, where the tensor components are
ordered `(11, 12, 13, 22, 23, 33)`.
"""
struct TimeResolvedStressTensorAccumulant{
    Times<:AbstractVector{<:Real},
} <: Accumulant{DiracDelta{Times}}
    A::Array{ComplexF64,3}
end

amplitudes(acc::TimeResolvedStressTensorAccumulant) = acc.A

"""
    TimeResolvedStressTensorKernel

Alias for a [`FourierStressTensorKernel`](@ref) whose temporal weight is a
[`DiracDelta`](@ref).
"""
const TimeResolvedStressTensorKernel = FourierStressTensorKernel{T,W,K} where {
    T,
    W<:DiracDelta,
    K,
}

struct DiracDeltaModeWorkspace{Times<:AbstractVector{<:Real}} <:
       ModeWorkspace{DiracDelta{Times}} end

ModeWorkspace(::DiracDelta{Times}, ::Int) where {Times} =
    DiracDeltaModeWorkspace{Times}()

prepare_source_modes!(
    ::DiracDeltaModeWorkspace,
    ::DiracDelta,
    ::AbstractVector{<:Real},
    source,
) = nothing

function _alloc_accumulant(
    weight::DiracDelta{Times},
    ::StressTensorTerm,
    Nk::Int,
) where {Times}
    return TimeResolvedStressTensorAccumulant{Times}(
        zeros(ComplexF64, 6, Nk, length(weight.times)),
    )
end

@inline function add_instantaneous_kinetic!(
    A::Array{ComplexF64,3},
    q::Int,
    a::Int,
    phase::ComplexF64,
    τ::Float64,
    nᵢnⱼ::SVector{6,Float64},
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
)
    amp = marker_weight * (ΔV / 3.0) * v^3 * τ^3 * phase
    @inbounds for I in 1:6
        A[I, q, a] = muladd(amp, nᵢnⱼ[I], A[I, q, a])
    end
    return nothing
end

@inline function add_instantaneous_potential!(
    A::Array{ComplexF64,3},
    q::Int,
    a::Int,
    phase::ComplexF64,
    τ::Float64,
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
    k::Float64,
)
    amp = marker_weight * im * ΔV * v^2 * n3 * τ^2 * phase / k
    @inbounds begin
        A[1, q, a] += amp
        A[4, q, a] += amp
        A[6, q, a] += amp
    end
    return nothing
end

@inline function add_instantaneous_term!(
    A, ::KineticTerm, q, a, phase, τ, nᵢnⱼ, n3, wΩ, ΔV, v, k,
)
    return add_instantaneous_kinetic!(A, q, a, phase, τ, nᵢnⱼ, wΩ, ΔV, v)
end

@inline function add_instantaneous_term!(
    A, ::PotentialTerm, q, a, phase, τ, nᵢnⱼ, n3, wΩ, ΔV, v, k,
)
    return add_instantaneous_potential!(A, q, a, phase, τ, n3, wΩ, ΔV, v, k)
end

@inline function add_instantaneous_term!(
    A, ::TotalStressTerm, q, a, phase, τ, nᵢnⱼ, n3, wΩ, ΔV, v, k,
)
    add_instantaneous_kinetic!(A, q, a, phase, τ, nᵢnⱼ, wΩ, ΔV, v)
    add_instantaneous_potential!(A, q, a, phase, τ, n3, wΩ, ΔV, v, k)
    return nothing
end

function accumulate_ray!(
    acc::TimeResolvedStressTensorAccumulant{Times},
    kernel::FourierStressTensorKernel{T,DiracDelta{Times},K},
    source::EnvelopeSource,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    τ_stop::Float64,
) where {
    T<:StressTensorTerm,
    Times<:AbstractVector{<:Real},
    K<:AbstractVector{<:Real},
}
    times = kernel.weight.times
    first_time = searchsortedlast(times, source.time) + 1
    last_time = searchsortedlast(times, source.time + τ_stop)
    first_time > last_time && return nothing

    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]
    z0 = source.center[3]

    @inbounds for a in first_time:last_time
        τ = Float64(times[a] - source.time)
        z = z0 + kernel.v * n3 * τ
        for q in eachindex(kernel.ks)
            k = Float64(kernel.ks[q])
            phase = cis(-k * z)
            add_instantaneous_term!(
                acc.A, kernel.term, q, a, phase, τ, nᵢnⱼ, n3,
                wΩ, kernel.ΔV, kernel.v, k,
            )
        end
    end
    return nothing
end
