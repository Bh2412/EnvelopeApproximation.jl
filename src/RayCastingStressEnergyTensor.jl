"""
    RayCastingStressEnergyTensor

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
module RayCastingStressEnergyTensor

import EnvelopeApproximation: TemporalWeight, CosineWeight, ConstantWeight
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Bubble, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Periodic
using EnvelopeApproximation.EnvelopeAnalysis: append_periodic_bubbles!
import EnvelopeApproximation.StressEnergyTensorComponents: contribution_indices
using StaticArrays
using LinearAlgebra

abstract type StressTensorTerm end
abstract type Accumulant{W<:TemporalWeight} end

struct KineticTerm <: StressTensorTerm end
struct PotentialTerm <: StressTensorTerm end
struct TotalStressTerm <: StressTensorTerm end

struct CosineAccumulant <: Accumulant{CosineWeight}
    A_plus::Matrix{ComplexF64}
    A_minus::Matrix{ComplexF64}
end

struct ConstantAccumulant <: Accumulant{ConstantWeight}
    A::Matrix{ComplexF64}
end

include("RayCastingStressEnergyTensor/SphericalQuadratureScheme.jl")
include("RayCastingStressEnergyTensor/CollisionSearch.jl")
include("RayCastingStressEnergyTensor/I2Kernels.jl")
include("RayCastingStressEnergyTensor/I3Kernels.jl")
include("RayCastingStressEnergyTensor/ModeAccumulation.jl")
include("RayCastingStressEnergyTensor/Strategies.jl")


# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ══════════════════════════════════
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
       ModeWorkspace,
       CosineModeWorkspace,
       ConstantModeWorkspace,
       amplitudes,
       ray_T_ij

# ═══════════════════════════════════════════════════════════════════════════════
# Ray-Casting T_ij Core Computation
# ═══════════════════════════════════════════════════════════════════════════════

allocate_accumulant(::CosineWeight, ::StressTensorTerm, Nk::Int) =
    CosineAccumulant(zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk))

allocate_accumulant(::ConstantWeight, ::StressTensorTerm, Nk::Int) =
    ConstantAccumulant(zeros(ComplexF64, 6, Nk))

amplitudes(acc::CosineAccumulant) = (acc.A_plus, acc.A_minus)
amplitudes(acc::ConstantAccumulant) = acc.A

check_ks(::KineticTerm, ks) = nothing
function check_ks(::Union{PotentialTerm, TotalStressTerm}, ks)
    any(iszero, ks) && throw(ArgumentError("k = 0 is not supported for PotentialTerm/TotalStressTerm"))
end
check_ks(terms::Tuple, ks) = foreach(t -> check_ks(t, ks), terms)

function build_periodic_blockers(snapshot::BubblesSnapShot, space::BoxSpace,
                                  v::Float64, t_end::Float64)
    original_times = map(nuc -> nuc[:time], snapshot.nucleations)
    original_centers = map(nuc -> nuc[:site].coordinates, snapshot.nucleations)

    bubbles = Bubble[]
    sizehint!(bubbles, length(original_centers))

    D_ray_max = v * t_end
    for i in eachindex(original_centers)
        R_b_max = v * (t_end - original_times[i])
        R_pad = D_ray_max + R_b_max
        push!(bubbles, Bubble(Point3(original_centers[i]), R_pad))
    end

    origin_map = Int[]
    padded_bubbles = append_periodic_bubbles!(bubbles, space, origin_map)

    padded_times = original_times[origin_map]
    return map((t, b) -> (time=t, site=b.center), padded_times, padded_bubbles)
end

function nucleation_ray_T_ij_contribution!(
    accumulants::Tuple,
    weight::W,
    terms::Tuple,
    ks::AbstractVector{<:Real},
    source_nucleation_idx::Int,
    source_nucleation::Nucleation,
    blockers_soa::NucleationSoA,
    collision_ws::SourceCollisionWorkspace,
    mode_ws::ModeWorkspace{W},
    markers::AbstractVector{SphericalQuadratureMarker},
    t_end::Float64;
    ΔV::Float64 = 1.0,
    v::Float64 = 1.0,
) where {W<:TemporalWeight}
    t_i = source_nucleation[:time]

    prepare_source_collision!(
        collision_ws, blockers_soa,
        source_nucleation, source_nucleation_idx, t_end, v,
    )

    prepare_source_modes!(mode_ws, weight, ks, source_nucleation)

    for marker in markers
        n̂ = marker.n̂

        τ_stop = find_collision_time(n̂, collision_ws, t_i, t_end, v)
        τ_stop < 1.0e-12 && continue

        for i in eachindex(terms)
            accumulate_marker_modes!(
                accumulants[i],
                weight,
                terms[i],
                ks,
                τ_stop,
                n̂,
                marker.weight,
                mode_ws,
                ΔV,
                v,
            )
        end
    end

    return nothing
end

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
                  space::BoxSpace, ::Periodic,
                  terms::Tuple,
                  weight::W,
                  quadrature::SphericalQuadratureScheme;
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:) where {W<:TemporalWeight}
    ks_f = ks isa AbstractRange ? ks : collect(Float64, ks)
    check_ks(terms, ks_f)
    Nk = length(ks_f)
    accumulants = map(t -> allocate_accumulant(weight, t, Nk), terms)

    isempty(snapshot.nucleations) && return accumulants

    markers = get_markers(quadrature)
    t_end = snapshot.t
    blocker_nucleations = build_periodic_blockers(snapshot, space, v, t_end)
    contributing_indices = contribution_indices(length(snapshot.nucleations), bubble_indices)

    blockers_soa = NucleationSoA(blocker_nucleations)
    collision_ws = SourceCollisionWorkspace(length(blocker_nucleations))
    mode_ws = ModeWorkspace(weight, Nk)

    for idx in contributing_indices
        nuc = snapshot.nucleations[idx]

        nucleation_ray_T_ij_contribution!(
            accumulants, weight, terms, ks_f,
            idx, nuc, blockers_soa, collision_ws, mode_ws, markers, t_end;
            ΔV=ΔV, v=v,
        )
    end

    return accumulants
end

function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, bc::Periodic,
                  named_terms::Tuple{Vararg{Pair{Symbol,<:StressTensorTerm}}},
                  weight::W,
                  quadrature::SphericalQuadratureScheme;
                  kwargs...) where {W<:TemporalWeight   }
    names = map(first, named_terms)
    terms = map(last, named_terms)
    accs = ray_T_ij(ks, snapshot, space, bc, terms, weight, quadrature; kwargs...)
    return NamedTuple{names}(accs)
end

end # module RayCastingStressEnergyTensor
