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

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ══════════════════════════════════
export SphericalQuadratureMarker, SphericalQuadratureScheme,
       RayCastingT_ij_CosineWeight,
       I3, collision_time, find_collision_time,
       UniformSphericalCapScheme, get_markers,
       ray_T_ij

# ═══════════════════════════════════════════════════════════════════════════════
# Ray-Casting T_ij Core Computation
# ═══════════════════════════════════════════════════════════════════════════════

function nucleation_ray_T_ij_contribution!(ks::Vector{Float64},
                                           source_nucleation_idx::Int,
                                           source_nucleation::Nucleation,
                                           blockers_soa::NucleationSoA,
                                           ws::SourceCollisionWorkspace,
                                           markers::Vector{SphericalQuadratureMarker},
                                           t_end::Float64;
                                           ΔV::Float64=1.0, v::Float64=1., A_plus::Matrix{ComplexF64},
                                           A_minus::Matrix{ComplexF64})
    t_i      = source_nucleation[:time]
    center_i = source_nucleation[:site].coordinates
    xi, yi, zi = center_i[1], center_i[2], center_i[3]

    prepare_source_collision!(ws, blockers_soa, xi, yi, zi, t_i, source_nucleation_idx, t_end, v)

    amp_base = (ΔV / 3.0) * v^3

    phase_plus  = Vector{ComplexF64}(undef, length(ks))
    phase_minus = Vector{ComplexF64}(undef, length(ks))

    @inbounds for q in eachindex(ks)
        k = ks[q]
        phase_plus[q]  = amp_base * cis( k * ( t_i - zi))
        phase_minus[q] = amp_base * cis(-k * ( t_i + zi))
    end

    for marker in markers
        n̂ = marker.n̂
        w = marker.weight
        n1, n2, n3 = n̂[1], n̂[2], n̂[3]

        τ_stop = find_collision_time(n1, n2, n3, ws, t_i, t_end, v)

        τ_stop < 1.0e-12 && continue

        for (k_idx, k) in enumerate(ks)
            α_plus  = k * ( 1.0 - v * n3)
            α_minus = k * (-1.0 - v * n3)

            I3_plus  = I3(α_plus,  0.0, τ_stop)
            I3_minus = I3(α_minus, 0.0, τ_stop)

            amp_plus = w *  phase_plus[k_idx]  * I3_plus
            amp_minus = w * phase_minus[k_idx] * I3_minus

            A_plus[1, k_idx] += amp_plus  * n1 * n1
            A_plus[2, k_idx] += amp_plus  * n1 * n2
            A_plus[3, k_idx] += amp_plus  * n1 * n3
            A_plus[4, k_idx] += amp_plus  * n2 * n2
            A_plus[5, k_idx] += amp_plus  * n2 * n3
            A_plus[6, k_idx] += amp_plus  * n3 * n3

            A_minus[1, k_idx] += amp_minus * n1 * n1
            A_minus[2, k_idx] += amp_minus * n1 * n2
            A_minus[3, k_idx] += amp_minus * n1 * n3
            A_minus[4, k_idx] += amp_minus * n2 * n2
            A_minus[5, k_idx] += amp_minus * n2 * n3
            A_minus[6, k_idx] += amp_minus * n3 * n3
        end
    end
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
function ray_T_ij(ks::AbstractVector{Float64}, snapshot::BubblesSnapShot,
                  space::BoxSpace, boundary_condition::Periodic,
                  strategy::RayCastingT_ij_CosineWeight;
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}
    ks_f = collect(Float64, ks)
    Nk = length(ks_f)

    empty_result = (zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk))
    isempty(snapshot.nucleations) && return empty_result

    A_plus  = zeros(ComplexF64, 6, Nk)
    A_minus = zeros(ComplexF64, 6, Nk)
    markers = strategy.markers

    t_end = snapshot.t
    full_nucleations = periodic_nucleations(snapshot.nucleations, v, 0.0, t_end, space)
    contributing_indices = contribution_indices(length(snapshot.nucleations), bubble_indices)

    blockers_soa = NucleationSoA(full_nucleations)
    ws = SourceCollisionWorkspace(length(full_nucleations))

    for idx in contributing_indices
        nuc = full_nucleations[idx]
        nucleation_ray_T_ij_contribution!(ks_f, idx, nuc, blockers_soa, ws, markers, t_end; ΔV, v, A_plus=A_plus, A_minus=A_minus)
    end
    
    return A_plus, A_minus
end

end # module RayCastingStressEnergyTensor
