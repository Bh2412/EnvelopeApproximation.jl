# Public ray-casting stress-tensor API.

using .BubblesEvolution: BubblesSnapShot
using .Spaces: BoxSpace
using .BoundaryConditions: Periodic
using .RayCastingEnvelopeIntegration:
    build_envelope_context,
    envelope_sources,
    envelope_integral,
    SphericalQuadratureScheme,
    UniformSphericalCapScheme,
    get_markers
using .StressTensor:
    StressTensorTerm,
    KineticTerm,
    PotentialTerm,
    TotalStressTerm,
    Accumulant,
    TimeResolvedStressTensorAccumulant,
    FourierStressTensorKernel,
    TimeResolvedStressTensorKernel,
    check_ks

"""
    ray_T_ij(ks, snapshot, space, boundary_condition;
             term=KineticTerm(), weight=CosineWeight(),
             quadrature=UniformSphericalCapScheme(16, 32),
             ΔV=1.0, v=1.0, bubble_indices=:)

Compute ray-cast time-integrated stress-tensor amplitudes.

- `term`: local contribution, e.g. `KineticTerm`, `PotentialTerm`, `TotalStressTerm`
- `weight`: time weighting, e.g. `CosineWeight`, `ConstantWeight`, or
  `ComplexExponential(ωs)`
- `quadrature`: ray directions

Returns an `Accumulant`. Use `amplitudes(acc)` to access the stored arrays.
For `ComplexExponential(ωs)`, the amplitude shape is
`(6, length(ks), length(ωs))`, ordered as tensor component, spatial mode, and
temporal frequency.
"""
function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, boundary_condition::Periodic;
                  term::StressTensorTerm=KineticTerm(),
                  weight::TemporalWeight=CosineWeight(),
                  quadrature::SphericalQuadratureScheme=UniformSphericalCapScheme(16, 32),
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:)
    return ray_T_ij(
        ks, snapshot, space, boundary_condition,
        term, weight, quadrature;
        ΔV=ΔV, v=v, bubble_indices=bubble_indices,
    )
end

function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, bc::Periodic,
                  term::StressTensorTerm,
                  weight::W,
                  quadrature::SphericalQuadratureScheme;
                  kwargs...) where {W<:TemporalWeight}
    return only(ray_T_ij(ks, snapshot, space, bc, (term,), weight, quadrature; kwargs...))
end

function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, bc::Periodic,
                  terms::Tuple,
                  weight::W,
                  quadrature::SphericalQuadratureScheme;
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:) where {W<:TemporalWeight}
    ks_f = ks isa AbstractRange ? ks : collect(Float64, ks)
    kernels = map(terms) do term
        FourierStressTensorKernel(term, weight, ks_f; ΔV=ΔV, v=v)
    end

    isempty(snapshot.nucleations) && return map(allocate_accumulant, kernels)

    context = build_envelope_context(snapshot, space, bc; v=v)
    sources  = envelope_sources(snapshot; bubble_indices=bubble_indices)
    markers  = get_markers(quadrature)

    return envelope_integral(kernels, sources, context, markers; τ_min=1.0e-12)
end

function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, bc::Periodic,
                  named_terms::Tuple{Vararg{Pair{Symbol,<:StressTensorTerm}}},
                  weight::W,
                  quadrature::SphericalQuadratureScheme;
                  kwargs...) where {W<:TemporalWeight}
    names = map(first, named_terms)
    terms = map(last, named_terms)
    accs  = ray_T_ij(ks, snapshot, space, bc, terms, weight, quadrature; kwargs...)
    return NamedTuple{names}(accs)
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

    context = build_envelope_context(snapshot, space, bc; v=v)
    sources = envelope_sources(snapshot; bubble_indices=bubble_indices)
    markers = get_markers(quadrature)

    return only(envelope_integral(
        (kernel,), sources, context, markers; τ_min=1.0e-12,
    ))
end

export ray_T_ij, ray_T_ij_at_times
