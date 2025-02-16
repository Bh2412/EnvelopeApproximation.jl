module VolumeVolume

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor: potential_integral
import EnvelopeApproximation.BubblesEvolution: BallSpace
import EnvelopeApproximation.ChebyshevCFT: First3MomentsChebyshevPlan
import EnvelopeApproximation.ISWPowerSpectrum: V, *, TopHemisphereLowerLeft, TopHemisphereUpperRight, align_ẑ, n̂
using HCubature
using StaticArrays
using QuadGK

function volume_ψ(ks:: AbstractVector{Float64}, 
                  snapshot:: BubblesSnapShot,
                  ball_space:: BallSpace,
                  chebyshev_plan:: First3MomentsChebyshevPlan{N},
                  _Δ:: Δ;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1.,
                  G:: Float64 = 1., 
                  kwargs...):: Vector{ComplexF64} where N
    t = snapshot.t
    c = 4π * (a ^ 2) * G
    f(τ:: Float64):: Vector{ComplexF64} = c * potential_integral(ks, current_bubbles(snapshot, τ), ball_space, 
                                                                 chebyshev_plan, _Δ; ΔV=ΔV) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

function integrand(ks:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot,
                   ball_space:: BallSpace,
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                   G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    rot = align_ẑ(n̂(ΦΘ))
    θ = ΦΘ[2]
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return @. abs2($volume_ψ(ks, _snap, ball_space, chebyshev_plan, _Δ; 
                             ΔV=ΔV, a=a, G=G, kwargs...)) * (sin(θ) / 2π)
end

export P

function P(ks:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
           ball_space:: BallSpace,
           chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
           _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
           G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    p(ΦΘ:: SVector{2, Float64}):: Vector{Float64} = integrand(ks, ΦΘ, snapshot, ball_space,
                                                              chebyshev_plan, _Δ; ΔV=ΔV,
                                                              a=a, G=G, kwargs...)
    return hcubature(p, TopHemisphereLowerLeft, TopHemisphereUpperRight; kwargs...)[1] ./ V(ball_space)
end

end