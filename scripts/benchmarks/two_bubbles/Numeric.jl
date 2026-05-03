begin
    using EnvelopeApproximation
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    using EnvelopeApproximation.Spaces
    using EnvelopeApproximation.BoundaryConditions
    using EnvelopeApproximation.StressEnergyTensorComponents
    using EnvelopeApproximation.TwoPointStressEnergyTensorModule
    using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
    using EnvelopeApproximation.EnvelopeAnalysis: align_ẑ, symmetric_tensor_inverse_rotation
    using LinearAlgebra
end

"""
    NumericTwoPointTij(k̂, ks, r1, r2, d, ρ_vac=1.0, V=1.0)

Computes ⟨T_ij(k) T_lm(-k)⟩ / V for two bubbles of radii r1, r2 separated
by distance d along ẑ, evaluated at all magnitudes in `ks` along direction `k̂`.

The system is rotated so that `k̂` aligns with ẑ (azimuthal reduction), the full
`ks` vector is passed to `TwoPointStressEnergyTensor` in a single call, then each
6×6 slice is rotated back to the original frame via
D = symmetric_tensor_inverse_rotation(align_ẑ(k̂)).

Returns a 6×6×Nk Array{ComplexF64, 3} in the symmetric tensor basis
(xx, xy, xz, yy, yz, zz).
"""
function NumericTwoPointTij(k̂::AbstractVector{T}, ks::AbstractVector{<:Real},
                             r1::T, r2::T, d::T,
                             ρ_vac::T=one(T), V::T=one(T))::Array{ComplexF64, 3} where T<:Real
    R = align_ẑ(Vec3(Float64.(k̂)))

    t    = max(Float64(r1), Float64(r2))
    nuc1 = (time = t - Float64(r1), site = Point3(0., 0., -Float64(d) / 2))
    nuc2 = (time = t - Float64(r2), site = Point3(0., 0.,  Float64(d) / 2))
    snapshot = R * BubblesSnapShot(Nucleation[nuc1, nuc2], t)

    space    = BoxSpace(cbrt(Float64(V)))
    plan     = VectorQuadGKPlan{7}()
    strategy = T_ij_AzimuthalReduction(plan, T_i_j_kernel(2))

    result_rotated = TwoPointStressEnergyTensor(t, t, collect(Float64, ks), snapshot, space,
                                                Vacuum(), strategy; ΔV = Float64(ρ_vac))
    D  = symmetric_tensor_inverse_rotation(R)
    Dt = D'
    return stack(D * result_rotated[:, :, ki] * Dt for ki in axes(result_rotated, 3))
end
