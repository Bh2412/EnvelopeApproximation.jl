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

"""
    IntegratedTwoPointStressEnergyTensor(ωs, snapshot, space, boundary_condition, strategy;
                                         ΔV=1.0, rtol=1e-3, atol=0.0, maxevals=typemax(Int))

Computes the time-integrated two-point correlator with weighting cos(k(t1−t2)):

  ∫∫ cos(k(t1−t2)) ⟨T_{ij}(k,t1) T_{lm}(-k,t2)⟩ dt1 dt2

using the factorisation cos(k Δt) = (e^{ikΔt} + e^{-ikΔt})/2, which reduces the 2D
integral to two 1D integrals:

  = (1/2V) [ A⁺_{ij}(k) conj(A⁺_{lm}(k)) + A⁻_{ij}(k) conj(A⁻_{lm}(k)) ]

where  A±_{ij}(k) = ∫ e^{±ikt} T_{ij}(k,t) dt.

Both integrals share a single QuadGK pass: T_{ij}(k,t) is evaluated once per quadrature
point and multiplied by e^{+ikt} and e^{-ikt} simultaneously.

Returns a 6×6×Nk array of ComplexF64.
"""
function IntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::TwoPointAzimuthalReduction;
    ΔV::Float64 = 1.0,
    rtol::Real = 1e-3,
    atol::Real = 0.0,
    maxevals::Int = typemax(Int)
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    V_inv   = 1.0 / volume(space)

    # Single quadrature pass: returns [vec(e^{+ikt} T); vec(e^{-ikt} T)], length 12Nk
    function integrand(t::Float64)
        bubbles = current_bubbles(snapshot, t)
        T = ∂iϕ∂jϕ(ωs_f, bubbles, space, boundary_condition,
                     strategy.plan, strategy.buffer; ΔV=ΔV)
        phases = cis.(ωs_f .* t)                          # e^{+ikt} per k
        return vcat(vec(T .* reshape(phases,        1, Nk)),
                    vec(T .* reshape(conj.(phases), 1, Nk)))
    end

    combined, _ = quadgk(integrand, t_start, t_end;
                         rtol=rtol, atol=atol, maxevals=maxevals)

    A_plus  = reshape(combined[1:6Nk],     6, Nk)   # ∫ e^{+ikt} T_{ij}(k,t) dt
    A_minus = reshape(combined[6Nk+1:end], 6, Nk)   # ∫ e^{-ikt} T_{ij}(k,t) dt

    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        ap = A_plus[:,  ki]
        am = A_minus[:, ki]
        # ap * ap' computes ap[i]*conj(ap[j]) — the outer product A⁺_{ij} conj(A⁺_{lm})
        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

end
