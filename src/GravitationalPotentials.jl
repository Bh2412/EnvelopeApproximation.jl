module GravitationalPotentials
 
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor: Ŋ
import EnvelopeApproximation.GeometricStressEnergyTensor: bubble_Ŋ_contribution!, k̂ik̂j∂_iφ∂_jφ
import EnvelopeApproximation.ChebyshevCFT: First3MomentsChebyshevPlan
import EnvelopeApproximation.BubbleBasics: Point3, coordinates, Vec3
import EnvelopeApproximation.BubblesEvolution: BallSpace
using QuadGK
using StaticArrays
import LinearAlgebra: norm

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  t:: Float64;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1., 
                  krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                  kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    bubbles = current_bubbles(snapshot, t)
    return (4π * a^2 * G) .* k̂ik̂jTij(ks, bubbles; krotations=krotations, ΔV=ΔV)
end

export ψ

function ψ(ks:: Vector{Vec3}, 
           snapshot:: BubblesSnapShot, 
           t:: Float64;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1., 
           G:: Float64 = 1., 
           krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
           kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, snapshot, τ;
                                                   ΔV=ΔV, a=a, G=G, 
                                                   krotations=krotations, 
                                                   kwargs...) * (t - τ)
    return quadgk(f, 0., t; kwargs...)[1]
end

export ψ

function Ŋ(ks:: Vector{Vec3}, 
           bubbles:: Bubbles;
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    c = @. (-12π * G * a ^ 2) / (ks ⋅ ks)
    return @. c * $ŋ_source(ks, bubbles; ΔV=ΔV, kwargs...)
end

export Ŋ

function Φ(ŋ:: Vector{ComplexF64}, Ψ:: Vector{ComplexF64}):: Vector{ComplexF64}
    return ŋ - Ψ
end

function ΦminusΨ(ks:: Vector{Vec3}, 
                 snapshot:: BubblesSnapShot, 
                 t:: Float64;
                 ΔV:: Float64 = 1., 
                 a:: Float64 = 1., 
                 G:: Float64 = 1., 
                 krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                 kwargs...)
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    _ψ = ψ(ks, snapshot, t; ΔV=ΔV, a=a, G=G, krotations=krotations, 
           kwargs...)
    _ŋ = Ŋ(ks, current_bubbles(snapshot, t); ΔV=ΔV, a=a, G=G, kwargs...)
    return @. _ŋ - 2 * _ψ
end

function ψ_source(ks:: AbstractVector{Float64}, 
                  bubbles:: AbstractVector{Bubble}, 
                  chebyshev_plan:: First3MomentsChebyshevPlan{N},
                  _Δ:: Δ;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1.,
                  G:: Float64 = 1.) where N
    return (4π * a ^ 2 * G) * k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)                  
end

function ψ_source(ks:: AbstractVector{Float64}, 
                  bubbles:: AbstractVector{Bubble},
                  ball_space:: BallSpace, 
                  chebyshev_plan:: First3MomentsChebyshevPlan{N},
                  _Δ:: Δ;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1.,
                  G:: Float64 = 1.) where N
    return (4π * a ^ 2 * G) * k̂ik̂jTij(ks, bubbles, ball_space, chebyshev_plan, _Δ; ΔV=ΔV)                  
end

function ψ(ks:: AbstractVector{Float64}, 
           snapshot:: BubblesSnapShot,
           chebyshev_plan:: First3MomentsChebyshevPlan{N},
           _Δ:: Δ;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1.,
           G:: Float64 = 1., 
           kwargs...) where N
    t = snapshot.t
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, current_bubbles(snapshot, τ), 
                                                   chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

function ψ(ks:: AbstractVector{Float64}, 
           snapshot:: BubblesSnapShot,
           ball_space:: BallSpace,
           chebyshev_plan:: First3MomentsChebyshevPlan{N},
           _Δ:: Δ;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1.,
           G:: Float64 = 1., 
           kwargs...) where N
    t = snapshot.t
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, current_bubbles(snapshot, τ), ball_space,
                                                   chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

function Ŋ(ks:: AbstractVector{Float64}, 
           bubbles:: AbstractVector{Bubble}, 
           chebyshev_plan:: First3MomentsChebyshevPlan{N},
           _Δ:: Δ;
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1.) where N
    V = zeros(ComplexF64, length(ks))
    domes = intersection_domes(bubbles)
    @inbounds for (bubble_index, _domes) in domes
        bubble_Ŋ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                               chebyshev_plan, _Δ; ΔV=ΔV)
    end
    c = -12π * G * a ^ 2
    return @. V * c / (ks ^ 2)
end

function surface_ψ_source(ks:: AbstractVector{Float64}, 
                          bubbles:: AbstractVector{Bubble}, 
                          chebyshev_plan:: First3MomentsChebyshevPlan{N},
                          _Δ:: Δ;
                          ΔV:: Float64 = 1., 
                          a:: Float64 = 1.,
                          G:: Float64 = 1.) where N
    return (4π * a ^ 2 * G) * k̂ik̂j∂_iφ∂_jφ(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)                  
end

function surface_ψ_source(ks:: AbstractVector{Float64}, 
                          bubbles:: AbstractVector{Bubble}, 
                          ball_space:: BallSpace,
                          chebyshev_plan:: First3MomentsChebyshevPlan{N},
                          _Δ:: Δ;
                          ΔV:: Float64 = 1., 
                          a:: Float64 = 1.,
                          G:: Float64 = 1.) where N
return (4π * a ^ 2 * G) * k̂ik̂j∂_iφ∂_jφ(ks, bubbles, ball_space, chebyshev_plan, _Δ; ΔV=ΔV)                  
end

function surface_ψ(ks:: AbstractVector{Float64}, 
                   snapshot:: BubblesSnapShot,
                   chebyshev_plan:: First3MomentsChebyshevPlan{N},
                   _Δ:: Δ;
                   ΔV:: Float64 = 1., 
                   a:: Float64 = 1.,
                   G:: Float64 = 1., 
                   kwargs...) where N
    t = snapshot.t
    f(τ:: Float64):: Vector{ComplexF64} = surface_ψ_source(ks, current_bubbles(snapshot, τ), 
                                                           chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

function surface_ψ(ks:: AbstractVector{Float64}, 
                   snapshot:: BubblesSnapShot,
                   ball_space:: BallSpace,
                   chebyshev_plan:: First3MomentsChebyshevPlan{N},
                   _Δ:: Δ;
                   ΔV:: Float64 = 1., 
                   a:: Float64 = 1.,
                   G:: Float64 = 1., 
                   kwargs...) where N
    t = snapshot.t
    f(τ:: Float64):: Vector{ComplexF64} = surface_ψ_source(ks, current_bubbles(snapshot, τ), ball_space, 
                                                           chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

end