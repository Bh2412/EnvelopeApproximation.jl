module GravitationalWaves

using EnvelopeApproximation.BubbleBasics: Bubble, Vec3
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor: ring_domes_complement_intersection!, _buffers, PeriodicInterval, polar_limits, IntersectionDome, intersection_domes
import IterTools: partition
import EnvelopeApproximation.ChebyshevCFT: VectorChebyshevPlanWithAtol, chebyshev_coeffs!, scale, translation, fourier_modes as chebyshev_fourier_modes, VectorChebyshevPlan
import EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan, fourier_modes as quadgk_fourier_modes
import EnvelopeApproximation.BubblesEvolution: BallSpace
import EnvelopeApproximation.ISWPowerSpectrum: n̂, align_ẑ
using StaticArrays
using HCubature

@inline function ∫_ϕ_x̂_ix̂_j(μ::Float64, p::PeriodicInterval)
    ϕ₁, ϕ₂ = p.ϕ1, p.ϕ1 + p.Δ
    
    # Pre-calculate powers and roots
    μ² = μ^2
    s² = 1 - μ²
    s  = sqrt(s²)
    
    # Pre-calculate trigonometric differences
    # Note: For cosine terms, the integral of sin is -cos, 
    # resulting in the order cos(1) - cos(2).
    Δϕ     = ϕ₂ - ϕ₁
    Δsinϕ  = sin(ϕ₂)  - sin(ϕ₁)
    Δcosϕ  = cos(ϕ₁)  - cos(ϕ₂)
    Δsin2ϕ = sin(2ϕ₂) - sin(2ϕ₁)
    Δcos2ϕ = cos(2ϕ₁) - cos(2ϕ₂) 
    
    # 6 components: xx, xy, xz, yy, yz, zz
    return SVector{6, Float64}(
        s² * (0.5Δϕ + 0.25Δsin2ϕ), # xx
        s² * 0.25Δcos2ϕ,           # xy
        μ*s * Δsinϕ,               # xz
        s² * (0.5Δϕ - 0.25Δsin2ϕ), # yy
        μ*s * Δcosϕ,               # yz
        μ² * Δϕ                    # zz
    )
end

struct x̂_ix̂_j
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

x̂_ix̂_j(n:: Int64) = x̂_ix̂_j(_buffers(n)...)

function (f:: x̂_ix̂_j)(μ:: Float64, bubble:: Bubble, 
                      intersection_domes:: Vector{IntersectionDome}):: SVector{6, Float64}
    V = SVector{6, Float64}(0., 0., 0., 0., 0., 0.)

    periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes, 
                                                             f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V += ∫_ϕ_x̂_ix̂_j(μ, interval)
    end
    return V
end

function fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, plan:: VectorQuadGKPlan{K}):: Matrix{ComplexF64} where {K}
    return quadgk_fourier_modes(f, ks, a, b, plan)
end

function fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, plan:: VectorChebyshevPlan{N, K}):: Matrix{ComplexF64} where {N, K}
    return collect(transpose(chebyshev_fourier_modes(f, ks, a, b, plan)))
end

function fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, plan:: VectorChebyshevPlanWithAtol{N, K, P}):: Matrix{ComplexF64} where {N, K, P}
    return collect(transpose(chebyshev_fourier_modes(f, ks, a, b, plan)[1]))
end

function bubble_∂iϕ∂jϕ_contribution!(V:: AbstractMatrix{ComplexF64},
                                     ks:: AbstractVector{Float64}, 
                                     bubble:: Bubble, 
                                     domes:: Vector{IntersectionDome}, 
                                     plan::P, 
                                     _x̂_ix̂_j:: x̂_ix̂_j; 
                                     ΔV:: Float64 = 1.) where {P}
    @assert size(V) == (length(ks), 6) "The output vector must be of the same length of the input k vector"
    modes = fourier_modes(μ -> _x̂_ix̂_j(μ, bubble, domes), ks * bubble.radius, -1., 1., plan)[1]
    es = map(ks) do k
        cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3.)
    end
    @. V += $reshape(es, $length(ks), 1) * modes    
end


function ∂iϕ∂jϕ(ks:: AbstractVector{Float64}, 
                bubbles:: AbstractVector{Bubble}, 
                ball_space:: BallSpace,
                plan:: P,
                _x̂_ix̂_j:: x̂_ix̂_j;
                ΔV:: Float64 = 1.):: Matrix{ComplexF64} where {P}
    V = zeros(ComplexF64, length(ks), 6)
    domes = intersection_domes(bubbles, ball_space)
    @inbounds for (bubble_index, _domes) in domes
    bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

"""
    symmetric_tensor_indices
    inverse_symmetric_tensor_indices

Mappings between the linear index (1-6) of a vector representation and the 
Cartesian indices (i, j) of a 3x3 symmetric tensor.

The mapping follows an upper-triangular, row-major ordering:
1 -> (1,1)  2 -> (1,2)  3 -> (1,3)
            4 -> (2,2)  5 -> (2,3)
                        6 -> (3,3)

# Usage
- `symmetric_tensor_indices[k]` returns the `(i, j)` tuple for linear index `k`.
- `inverse_symmetric_tensor_indices[(i, j)]` returns the linear index `k`.
"""
const symmetric_tensor_indices:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2=> (1, 2), 3=> (1, 3), 4 =>(2, 2), 5 =>(2, 3), 6 => (3, 3))
const inverse_symmetric_tensor_indices:: Dict{Tuple{Int, Int}, Int} = Dict(zip(values(symmetric_tensor_indices), keys(symmetric_tensor_indices)))

function symmetric_dot(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
    r = 0.
    for ĩ in 1:6
        (i, j) = symmetric_tensor_indices[ĩ]
        (i == j) && (r += (T1[ĩ])' * T2[ĩ]); continue
        r += 2 * (T1[ĩ])' * T2[ĩ]
    end
    return r
end

function δ(T:: AbstractVector{ComplexF64}):: ComplexF64
    return T[1] + T[4] + T[6]
end

function zz(T:: AbstractVector{ComplexF64})
    return T[6]
end

function Λ(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
    r = 0.
    r += symmetric_dot(T1, T2)
    r += (-2) * @views (T1[[3, 5, 6]]' * T2[[3, 5, 6]])
    zz1 = zz(T1)'
    zz2 = zz(T2)
    δ1 = δ(T1)'
    δ2 = δ(T2)
    r += (1. / 2) * zz1 * zz2
    r += (-1. / 2) * δ1 * δ2
    r += (1. / 2) * δ1 * zz2
    r += (1. / 2) * zz1 * δ2
    return r
end

function Λ(T:: AbstractVector{ComplexF64}):: Float64
    return Λ(T, T)
end

export Directional_Π

# Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
function Directional_Π(_n̂:: Vec3, t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
                       ball_space:: BallSpace, plan:: P, 
                       _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1.):: Vector{ComplexF64} where {P}
    _snap = align_ẑ(_n̂) * snapshot
    bubbles1 = current_bubbles(_snap, t1)
    bubbles2 = current_bubbles(_snap, t2)
    T1 = ∂iϕ∂jϕ(ωs, bubbles1, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV)
    T2 = ∂iϕ∂jϕ(ωs, bubbles2, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV)
    return @. Λ($eachcol(T1), $eachcol(T2)) / $volume(ball_space)
end
export Π

function Π(t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
           ball_space:: BallSpace, plan::P, 
           _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., kwargs...):: Tuple{Vector{ComplexF64}, Float64} where {P}
    function f(_n̂:: SVector{2, Float64}):: Vector{Float64}
        ϕ, θ = _n̂
        return @. 2 * real($Directional_Π($n̂(ϕ, θ), t1, t2, ωs, snapshot, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV) * $sin(θ))
    end
    v, err = hcubature(f, SVector(0., 0.,), SVector(2π, π / 2); kwargs...)  # It is enough to integrate over half the ski
    return v ./ 4π, err / 4π
end


end