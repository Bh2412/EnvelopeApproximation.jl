# Instantaneous Fourier stress tensor on a prescribed time grid.

"""
    TimeResolvedStressTensorAccumulant

Accumulant returned by [`ray_T_ij_at_times`](@ref). `A` has axes
`(tensor_component, wave_number, time)`, where the tensor components are
ordered `(11, 12, 13, 22, 23, 33)`.
"""
struct TimeResolvedStressTensorAccumulant
    A::Array{ComplexF64,3}
end

amplitudes(acc::TimeResolvedStressTensorAccumulant) = acc.A

"""
    TimeResolvedStressTensorKernel(term, times, ks, ΔV, v)

Kernel for evaluating the Fourier-space stress tensor at the times in `times`.
For a ray nucleated at `tₙ`, only samples satisfying
`0 < t - tₙ ≤ τ_stop` contribute.
"""
struct TimeResolvedStressTensorKernel{
    T<:StressTensorTerm,
    Times<:AbstractVector,
    K<:AbstractVector,
} <: Kernel
    term::T
    times::Times
    ks::K
    ΔV::Float64
    v::Float64
end

function allocate_accumulant(kernel::TimeResolvedStressTensorKernel)
    return TimeResolvedStressTensorAccumulant(
        zeros(ComplexF64, 6, length(kernel.ks), length(kernel.times)),
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
    acc::TimeResolvedStressTensorAccumulant,
    kernel::TimeResolvedStressTensorKernel,
    source::LightConeSource,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    τ_stop::Float64,
)
    times = kernel.times
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

"""
    ray_T_ij_at_times(times, ks, snapshot, space, boundary_condition;
                      term=KineticTerm(),
                      quadrature=UniformSphericalCapScheme(16, 32),
                      ΔV=1.0, v=1.0, bubble_indices=:)

Evaluate the ray-cast Fourier stress tensor `T̃ᵢⱼ(t, k)` at a sorted time grid.
The returned [`TimeResolvedStressTensorAccumulant`](@ref) stores an array of
shape `(6, length(ks), length(times))`; use [`amplitudes`](@ref) to access it.

This is the delta-time kernel corresponding to the existing time-integrated
kernel. Consequently, integrating its amplitudes over `times` reproduces
`ray_T_ij(..., ConstantWeight(), ...)` (up to time-quadrature error).
"""
function ray_T_ij_at_times(
    times::AbstractVector{<:Real},
    ks::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    bc::Periodic;
    term::StressTensorTerm=KineticTerm(),
    quadrature::SphericalQuadratureScheme=UniformSphericalCapScheme(16, 32),
    ΔV::Float64=1.0,
    v::Float64=1.0,
    bubble_indices=:,
)
    issorted(times) || throw(ArgumentError("times must be sorted in nondecreasing order"))

    times_f = collect(Float64, times)
    ks_f = ks isa AbstractRange ? ks : collect(Float64, ks)
    check_ks(term, ks_f)

    kernel = TimeResolvedStressTensorKernel(term, times_f, ks_f, ΔV, v)
    isempty(snapshot.nucleations) && return allocate_accumulant(kernel)

    context = build_lightcone_context(snapshot, space, bc; v=v)
    sources = lightcone_sources(snapshot; bubble_indices=bubble_indices)
    markers = get_markers(quadrature)

    return only(integrate_lightcone_surfaces(
        (kernel,), sources, context, markers; τ_min=1.0e-12,
    ))
end
