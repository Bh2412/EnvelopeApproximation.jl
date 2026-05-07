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

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Bubble, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Periodic
using EnvelopeApproximation.EnvelopeAnalysis: append_periodic_bubbles!
import EnvelopeApproximation.StressEnergyTensorComponents: contribution_indices
using StaticArrays
using LinearAlgebra

include("RayCastingStressEnergyTensor/SphericalQuadratureScheme.jl")
include("RayCastingStressEnergyTensor/Strategies.jl")
include("RayCastingStressEnergyTensor/CollisionSearch.jl")
include("RayCastingStressEnergyTensor/I3Kernels.jl")
include("RayCastingStressEnergyTensor/PeriodicBlockers.jl")
include("RayCastingStressEnergyTensor/ModeAccumulation.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ══════════════════════════════════
export SphericalQuadratureScheme,
       RayCastingT_ij_CosineWeight,
       UniformSphericalCapScheme,
       ray_T_ij

# ═══════════════════════════════════════════════════════════════════════════════
# Ray-Casting T_ij Core Computation
# ═══════════════════════════════════════════════════════════════════════════════

function nucleation_ray_T_ij_contribution!(
    ks::AbstractVector{<:Real},
    source_nucleation_idx::Int,
    source_nucleation::Nucleation,
    blockers_soa::NucleationSoA,
    collision_ws::SourceCollisionWorkspace,
    mode_ws::SourceModeWorkspace,
    markers::AbstractVector{SphericalQuadratureMarker},
    t_end::Float64;
    ΔV::Float64 = 1.0,
    v::Float64 = 1.0,
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
)
    t_i = source_nucleation[:time]
    center_i = source_nucleation[:site].coordinates
    xi, yi, zi = center_i[1], center_i[2], center_i[3]

    prepare_source_collision!(
        collision_ws, blockers_soa,
        xi, yi, zi, t_i,
        source_nucleation_idx, t_end, v,
    )

    amp_base = (ΔV / 3.0) * v^3

    resize!(mode_ws, length(ks))
    prepare_source_phases!(mode_ws, ks, t_i, zi, amp_base)

    for marker in markers
        n̂ = marker.n̂

        τ_stop = find_collision_time(n̂, collision_ws, t_i, t_end, v)
        τ_stop < 1.0e-12 && continue

        accumulate_marker_modes!(
            A_plus,
            A_minus,
            ks,
            τ_stop,
            n̂,
            marker.weight,
            mode_ws,
            v,
        )
    end

    return nothing
end

"""
Compute the time-integrated amplitudes

  A±_I(k) = (ΔV/3) ∑_i ∑_a w_a n̂_{a,I} e^{-ikzᵢ} e^{±iktᵢ} v³
            I₃(k(±1-vn̂_z); 0, τ_stop)

where I labels the six symmetric tensor components
xx=1, xy=2, xz=3, yy=4, yz=5, zz=6.

Returns two 6×Nk matrices, `A_plus` and `A_minus`, whose columns correspond
to individual k values.

For each k index q, the cosine-weighted tensor correlator is the 6×6 matrix

  D[:, :, q] =
      (1/(2V)) * (
          A_plus[:, q]  * A_plus[:, q]' +
          A_minus[:, q] * A_minus[:, q]'
      )

where `'` denotes complex conjugate transpose.
"""
function ray_T_ij(ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot,
                  space::BoxSpace, boundary_condition::Periodic,
                  strategy::RayCastingT_ij_CosineWeight;
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}
    ks_f = ks isa AbstractRange ? ks : collect(Float64, ks)
    Nk = length(ks_f)

    if isempty(snapshot.nucleations)
        return zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk)
    end

    A_plus  = zeros(ComplexF64, 6, Nk)
    A_minus = zeros(ComplexF64, 6, Nk)
    markers = strategy.markers

    t_end = snapshot.t
    full_nucleations = periodic_nucleations(snapshot.nucleations, v, 0.0, t_end, space)
    contributing_indices = contribution_indices(length(snapshot.nucleations), bubble_indices)

    blockers_soa = NucleationSoA(full_nucleations)
    collision_ws = SourceCollisionWorkspace(length(full_nucleations))
    mode_ws = SourceModeWorkspace(Nk)

    for idx in contributing_indices
        nuc = full_nucleations[idx]

        nucleation_ray_T_ij_contribution!(
            ks_f,
            idx,
            nuc,
            blockers_soa,
            collision_ws,
            mode_ws,
            markers,
            t_end;
            ΔV = ΔV,
            v = v,
            A_plus = A_plus,
            A_minus = A_minus,
        )
    end
    
    return A_plus, A_minus
end

end # module RayCastingStressEnergyTensor
