"""
    StressTensor

Ray-casting method for computing T_ij using deterministic spherical quadrature
and analytic time integration via the I₃(α; a, b) formula.

Key features:
- No dome-ring topology complexity
- No time quadrature (all integrals are closed-form)
- Pluggable spherical quadrature scheme interface
- Direct collision detection along rays

# References
The method discretizes the angular integral using spherical quadrature:
  ∫ dΩ f(n̂) ≈ ∑_a w_a f(n̂_a)

For each ray marker n̂_a, collision times are computed and the time integral
is evaluated analytically:
  A_ij^±(k) = (ΔV/3) ∑_a w_a n̂_{a,i} n̂_{a,j} e^{-ikz_i} e^{±ikt_i} v³ I₃(α_±; τ_start, τ_stop)

where I₃ is the closed-form integral of τ³ exp(iατ) from a to b.
"""
module StressTensor

import EnvelopeApproximation: TemporalWeight, CosineWeight, ConstantWeight
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Periodic
using StaticArrays
using LinearAlgebra

import ..RayCastingEnvelopeIntegration: allocate_accumulant, prepare_kernel!, accumulate_ray!
using ..RayCastingEnvelopeIntegration:
    LightConeSource, build_lightcone_context, lightcone_sources, integrate_lightcone_surfaces,
    collision_time,
    SphericalQuadratureScheme, SphericalQuadratureMarker, UniformSphericalCapScheme, get_markers

abstract type StressTensorTerm end
abstract type Accumulant{W<:TemporalWeight} end

struct KineticTerm   <: StressTensorTerm end
struct PotentialTerm <: StressTensorTerm end
struct TotalStressTerm <: StressTensorTerm end

struct CosineAccumulant <: Accumulant{CosineWeight}
    A_plus::Matrix{ComplexF64}
    A_minus::Matrix{ComplexF64}
end

struct ConstantAccumulant <: Accumulant{ConstantWeight}
    A::Matrix{ComplexF64}
end

include("I2Kernels.jl")
include("I3Kernels.jl")
include("ModeAccumulation.jl")
include("TimeResolvedAccumulation.jl")
include("Strategies.jl")


# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════
export SphericalQuadratureScheme,
       RayCastingSphericalQuadrature,
       UniformSphericalCapScheme,
       StressTensorTerm,
       KineticTerm,
       PotentialTerm,
       TotalStressTerm,
       TemporalWeight,
       CosineWeight,
       ConstantWeight,
       Accumulant,
       CosineAccumulant,
       ConstantAccumulant,
       TimeResolvedStressTensorAccumulant,
       ModeWorkspace,
       CosineModeWorkspace,
       ConstantModeWorkspace,
       FourierStressTensorKernel,
       TimeResolvedStressTensorKernel,
       amplitudes,
       ray_T_ij,
       ray_T_ij_at_times

# ═══════════════════════════════════════════════════════════════════════════════
# Accumulant helpers
# ═══════════════════════════════════════════════════════════════════════════════

amplitudes(acc::CosineAccumulant) = (acc.A_plus, acc.A_minus)
amplitudes(acc::ConstantAccumulant) = acc.A

check_ks(::KineticTerm, ks) = nothing
function check_ks(::Union{PotentialTerm, TotalStressTerm}, ks)
    any(iszero, ks) && throw(ArgumentError("k = 0 is not supported for PotentialTerm/TotalStressTerm"))
end
check_ks(terms::Tuple, ks) = foreach(t -> check_ks(t, ks), terms)

# Internal allocators used by the early-return path and by allocate_accumulant(kernel).
_alloc_accumulant(::CosineWeight, ::StressTensorTerm, Nk::Int) =
    CosineAccumulant(zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk))

_alloc_accumulant(::ConstantWeight, ::StressTensorTerm, Nk::Int) =
    ConstantAccumulant(zeros(ComplexF64, 6, Nk))

# ═══════════════════════════════════════════════════════════════════════════════
# Fourier stress-tensor kernel
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FourierStressTensorKernel{T,W}

Kernel that accumulates Fourier-mode stress-tensor amplitudes inside the generic
`integrate_lightcone_surfaces` engine.

Packages the physics parameters that used to live inside `ray_T_ij`:
`term`, `weight`, `ks`, `mode_ws`, `ΔV`, `v`.

All kernels built for the same call share a single `mode_ws` instance because
they all write the same phases; sharing avoids redundant work.
"""
struct FourierStressTensorKernel{T<:StressTensorTerm, W<:TemporalWeight, K<:AbstractVector}
    term::T
    weight::W
    ks::K
    mode_ws::ModeWorkspace{W}
    ΔV::Float64
    v::Float64
end

# --- extensions of the RayCastingEnvelopeIntegration generic kernel interface ---

function allocate_accumulant(kernel::FourierStressTensorKernel)
    return _alloc_accumulant(kernel.weight, kernel.term, length(kernel.ks))
end

function prepare_kernel!(kernel::FourierStressTensorKernel, source::LightConeSource)
    prepare_source_modes!(kernel.mode_ws, kernel.weight, kernel.ks, source)
    return nothing
end

function accumulate_ray!(
    acc::Accumulant,
    kernel::FourierStressTensorKernel,
    ::LightConeSource,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    τ_stop::Float64,
)
    accumulate_marker_modes!(
        acc, kernel.weight, kernel.term, kernel.ks,
        τ_stop, n̂, wΩ, kernel.mode_ws, kernel.ΔV, kernel.v,
    )
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Ray-Casting T_ij — public API (signatures unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ray_T_ij(ks, snapshot, space, boundary_condition;
             term=KineticTerm(), weight=CosineWeight(),
             quadrature=UniformSphericalCapScheme(16, 32),
             ΔV=1.0, v=1.0, bubble_indices=:)

Compute ray-cast time-integrated stress-tensor amplitudes.

- `term`: local contribution, e.g. `KineticTerm`, `PotentialTerm`, `TotalStressTerm`
- `weight`: time weighting, e.g. `CosineWeight` or `ConstantWeight`
- `quadrature`: ray directions

Returns an `Accumulant`. Use `amplitudes(acc)` to access the stored arrays.
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
    check_ks(terms, ks_f)
    Nk = length(ks_f)

    isempty(snapshot.nucleations) && return map(t -> _alloc_accumulant(weight, t, Nk), terms)

    context = build_lightcone_context(snapshot, space, bc; v=v)
    sources  = lightcone_sources(snapshot; bubble_indices=bubble_indices)
    markers  = get_markers(quadrature)

    mode_ws = ModeWorkspace(weight, Nk)
    kernels = map(terms) do term
        FourierStressTensorKernel(term, weight, ks_f, mode_ws, ΔV, v)
    end

    return integrate_lightcone_surfaces(kernels, sources, context, markers; τ_min=1.0e-12)
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

end # module StressTensor
