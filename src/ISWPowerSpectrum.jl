module ISWPowerSpectrum
using HCubature
import Base.*
using StaticArrays
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.ChebyshevCFT: First3MomentsChebyshevPlan
import EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, BallSpace
import EnvelopeApproximation.GeometricStressEnergyTensor: align_ẑ, Δ
import EnvelopeApproximation.GravitationalPotentials: ψ, surface_ψ

*(rot:: SMatrix{3, 3, Float64}, p:: Point3):: Point3 = Point3(rot * p.coordinates)

V(ball_space:: BallSpace):: Float64 = (4π / 3) * (ball_space.radius ^ 3)

function *(rot:: SMatrix{3, 3, Float64}, snapshot:: BubblesSnapShot):: BubblesSnapShot
    new_nucleations = [(time=nuc.time, site=rot * nuc.site) for nuc in snapshot.nucleations]
    return BubblesSnapShot(new_nucleations, snapshot.t, snapshot.radial_profile)
end

function n̂(ϕ:: Float64, θ:: Float64):: Vec3
    return Vec3((sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))) 
end 

function n̂(x:: SVector{2, Float64}):: Vec3
    n̂(x[1], x[2]) 
end 

function integrand(ks:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                   G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    rot = align_ẑ(n̂(ΦΘ))
    θ = ΦΘ[2]
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return @. abs2($ψ(ks, _snap, chebyshev_plan, _Δ; 
                      ΔV=ΔV, a=a, G=G, kwargs...)) * (sin(θ) / 2π)
end

const TopHemisphereLowerLeft:: SVector{2, Float64} = SVector{2, Float64}(0., 0.)
const TopHemisphereUpperRight:: SVector{2, Float64} = SVector{2, Float64}(2π, π / 2)

export P

function P(ks:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
           chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
           _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
           G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    p(ΦΘ:: SVector{2, Float64}):: Vector{Float64} = integrand(ks, ΦΘ, snapshot, 
                                                                 chebyshev_plan, _Δ; ΔV=ΔV,
                                                                 a=a, G=G, kwargs...)
    return hcubature(p, TopHemisphereLowerLeft, TopHemisphereUpperRight; kwargs...)[1]
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
    return @. abs2($ψ(ks, _snap, ball_space, chebyshev_plan, _Δ; 
                      ΔV=ΔV, a=a, G=G, kwargs...)) * (sin(θ) / 2π)
end

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


function surface_integrand(ks:: AbstractVector{Float64}, 
                           ΦΘ:: SVector{2, Float64}, 
                           snapshot:: BubblesSnapShot, 
                           chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                           _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                           G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    rot = align_ẑ(n̂(ΦΘ))
    θ = ΦΘ[2]
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return @. abs2($surface_ψ(ks, _snap, chebyshev_plan, _Δ; 
                              ΔV=ΔV, a=a, G=G, kwargs...)) * (sin(θ) / 2π)
end

function surface_integrand(ks:: AbstractVector{Float64}, 
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
    return @. abs2($surface_ψ(ks, _snap, ball_space, chebyshev_plan, _Δ; 
                              ΔV=ΔV, a=a, G=G, kwargs...)) * (sin(θ) / 2π)
end

export surface_P

function surface_P(ks:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                    G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    p(ΦΘ:: SVector{2, Float64}):: Vector{ComplexF64} = surface_integrand(ks, ΦΘ, snapshot, 
                                                                         chebyshev_plan, _Δ; ΔV=ΔV,
                                                                         a=a, G=G, kwargs...)
    return hcubature(p, TopHemisphereLowerLeft, TopHemisphereUpperRight; kwargs...)[1]
end

function surface_P(ks:: AbstractVector{Float64}, snapshot:: BubblesSnapShot,
                   ball_space:: BallSpace, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                   G:: Float64 = 1., kwargs...):: Vector{Float64} where N
    p(ΦΘ:: SVector{2, Float64}):: Vector{ComplexF64} = surface_integrand(ks, ΦΘ, snapshot, ball_space, 
                                                                         chebyshev_plan, _Δ; ΔV=ΔV,
                                                                         a=a, G=G, kwargs...)
    return hcubature(p, TopHemisphereLowerLeft, TopHemisphereUpperRight; kwargs...)[1] ./ V(ball_space)
end

include("PowerSpectra/VolumeVolume.jl")

end