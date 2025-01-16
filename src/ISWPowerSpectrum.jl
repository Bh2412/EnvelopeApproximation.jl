module ISWPowerSpectrum
using HCubature
import Base.*
using StaticArrays
import EnvelopeApproximation.BubblesEvolution: BubblesSnapShot
import EnvelopeApproximation.GeometricStressEnergyTensor: align_ẑ, Δ
import EnvelopeApproximation.GravitationalPotentials: ψ

*(rot:: SMatrix{3, 3, Float64}, p:: Point3):: Point3 = Point3(rot * p.coordinates)

function *(rot:: SMatrix{3, 3, Float64}, snapshot:: BubblesSnapShot):: BubblesSnapShot
    new_nucleations = [(time=nuc.time, site=rot * nuc.site) for nuc in snapshot.nucleations]
    return BubblesSnapShot(new_nucleations, snapshot.t, snapshot.radial_profile)
end

function n̂(x:: SVector{2, Float64}):: Vec3
    ϕ, θ = x
    return Vec3((sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))) 
end 

function integrand(ks:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                   G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
    rot = align_ẑ(n̂(ΦΘ))
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return @. 8 .* abs2.(ψ(ks, _snap, chebyshev_plan, _Δ; 
                           ΔV=ΔV, a=a, G=G, kwargs...))
end

# @btime integrand(ks, SVector(0., π /2))
nd 

function integrand(ks:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _Δ:: Δ; ΔV:: Float64 = 1., a:: Float64 = 1.,
                   G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
    rot = align_ẑ(n̂(ΦΘ))
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return 2 * 4 .* abs2.(ψ(ks, _snap, chebyshev_plan, _Δ; 
                         ΔV=ΔV, a=a, G=G, kwargs...))
end
const TopHemisphereLowerLeft:: SVector{2, Float64} = SVector{2, Float64}(0., 0.)
const TopHemisphereUpperRight:: SVector{2, Float64} = SVector{2, Float64}(2π, π / 2)

function P(ks:: AbstractVector{Float64}):: Vector{Float64}
    p(ΦΘ:: SVector{2, Float64}):: Vector{ComplexF64} = integrand(ks, ΦΘ, snapshot, 
                                                                 chebyshev_plan, _Δ, ΔV=ΔV,
                                                                 a=a, G=G, kwargs...)
    return hcubature(p, TopHemisphereLowerLeft, TopHemisphereUpperRight; kwargs...)[1]
end

end