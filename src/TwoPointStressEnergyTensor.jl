module TwoPointStressEnergyTensorModule

using EnvelopeApproximation.StressEnergyTensorComponents
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, current_bubbles
using EnvelopeApproximation.Spaces: AbstractSpace, volume
using EnvelopeApproximation.BoundaryConditions: BoundaryCondition
using QuadGK

export TwoPointAzimuthalReduction, TwoPointStressEnergyTensor, IntegratedTwoPointStressEnergyTensor

"""
    TwoPointStressEnergyTensor(t1, t2, ωs, snapshot, space, boundary_condition, strategy; ΔV=1.0)

Computes the two-point stress-energy tensor correlator ⟨T_{ij}(k,t1) T_{lm}(-k,t2)⟩
for k along ẑ, returning a 6×6×Nk array of ComplexF64.

Both tensor indices follow the symmetric tensor convention: xx=1, xy=2, xz=3, yy=4, yz=5, zz=6.
Element [ij, lm, ki] = T1[ij, ki] * conj(T2[lm, ki]) / V.
"""
function TwoPointStressEnergyTensor(t1::Real, t2::Real, ωs::AbstractVector{<:Real},
                                     snapshot::BubblesSnapShot, space::AbstractSpace,
                                     boundary_condition::BoundaryCondition,
                                     strategy::TwoPointAzimuthalReduction;
                                     ΔV::Float64=1.0)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    bubbles1 = current_bubbles(snapshot, Float64(t1))
    bubbles2 = current_bubbles(snapshot, Float64(t2))
    T1 = ∂iϕ∂jϕ(ωs_f, bubbles1, space, boundary_condition, strategy.plan, strategy.buffer; ΔV=ΔV)
    T2 = ∂iϕ∂jϕ(ωs_f, bubbles2, space, boundary_condition, strategy.plan, strategy.buffer; ΔV=ΔV)
    V_inv = 1.0 / volume(space)
    return reshape(T1, 6, 1, :) .* conj.(reshape(T2, 1, 6, :)) .* V_inv
end

end
