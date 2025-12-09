module GravitationalWaves

using EnvelopeApproximation.BubbleBasics: Bubble, Vec3
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor: ring_domes_complement_intersection!, _buffers, PeriodicInterval, polar_limits, IntersectionDome, intersection_domes
import IterTools: partition
import EnvelopeApproximation.CFTInterface: CFTPlan, fourier_modes
using EnvelopeApproximation.ChebyshevCFT: VectorChebyshevPlanWithAtol, chebyshev_coeffs!, scale, translation, VectorChebyshevPlan
using EnvelopeApproximation.AngularIntegrationInterface: AbstractAngularIntegrationPlan, integrate_angular
using EnvelopeApproximation.SphericalHarmonics: SHPlan, integrate_angular
using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
import EnvelopeApproximation.BubblesEvolution: BallSpace
import EnvelopeApproximation.ISWPowerSpectrum: n̂, align_ẑ
using StaticArrays
using HCubature

export Directional_Π, Π, x̂_ix̂_j, symmetric_tensor_indices, inverse_symmetric_tensor_indices

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

mutable struct x̂_ix̂_j
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

# Computes Λᵢⱼₗₘx̂ₗx̂ₘ - only depends on Txx-Tyy, Txy
@inline function ∫_ϕ_Λx̂x̂(μ::Float64, p::PeriodicInterval)
    ϕ₁, ϕ₂ = p.ϕ1, p.ϕ1 + p.Δ
    
    # Pre-calculate powers and roots
    μ² = μ^2
    s² = 1 - μ²
    
    # Pre-calculate trigonometric differences
    # Note: For cosine terms, the integral of sin is -cos, 
    Δsin2ϕ = sin(2ϕ₂) - sin(2ϕ₁)
    Δcos2ϕ = cos(2ϕ₁) - cos(2ϕ₂) 
    
    # 2 components: 1/2 (xx - yy), xy
    return SVector{6, Float64}(
        s² * 0.25Δsin2ϕ, # xx
        s² * 0.25Δcos2ϕ, # xy
    )
end

mutable struct Λx̂x̂
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

Λx̂x̂(n:: Int64) = Λx̂x̂(_buffers(n)...)

function (f:: Λx̂x̂)(μ:: Float64, bubble:: Bubble, 
                   intersection_domes:: Vector{IntersectionDome}):: SVector{6, Float64}
    V = SVector{2, Float64}(0., 0.)
    periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes, 
                                                             f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V += ∫_ϕ_Λx̂x̂(μ, interval)
    end
    return V
end

function dispatch_fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, plan:: VectorQuadGKPlan{K}):: Matrix{ComplexF64} where {K}
    val, err = fourier_modes(f, ks, a, b, plan)
    return val
end

function dispatch_fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, plan:: VectorChebyshevPlan{N, K}):: Matrix{ComplexF64} where {N, K}
    return collect(permutedims(fourier_modes(f, ks, a, b, plan)))
end

function dispatch_fourier_modes(f, ks:: AbstractVector{Float64}, a:: Float64, b:: Float64, 
                                plan:: VectorChebyshevPlanWithAtol{N, K, P}):: Matrix{ComplexF64} where {N, K, P}
    return collect(permutedims(fourier_modes(f, ks, a, b, plan)[1]))
end

function bubble_∂iϕ∂jϕ_contribution!(V:: AbstractMatrix{ComplexF64},
                                     ks:: AbstractVector{Float64}, 
                                     bubble:: Bubble, 
                                     domes:: Vector{IntersectionDome}, 
                                     plan::CFTPlan, 
                                     _x̂_ix̂_j:: x̂_ix̂_j; 
                                     ΔV:: Float64 = 1.)
    @assert size(V) == (6, length(ks)) "Output buffer V must be (6, Nk)"
    modes = dispatch_fourier_modes(μ -> _x̂_ix̂_j(μ, bubble, domes), ks * bubble.radius, -1., 1., plan)
    es = map(ks) do k
        cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3.)
    end
    @. V += $reshape(es, 1, $length(ks)) * modes    
end

function bubble_Λ∂ᵢϕ∂ⱼϕ_contribution!(V:: AbstractMatrix{ComplexF64},
                                      ks:: AbstractVector{Float64}, 
                                      bubble:: Bubble, 
                                      domes:: Vector{IntersectionDome}, 
                                      plan::CFTPlan, 
                                      _Λx̂x̂:: Λx̂x̂; 
                                      ΔV:: Float64 = 1.)
    @assert size(V) == (6, length(ks)) "Output buffer V must be (6, Nk)"
    modes = dispatch_fourier_modes(μ -> _Λx̂x̂(μ, bubble, domes), ks * bubble.radius, -1., 1., plan)
    es = map(ks) do k
        cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3.)
    end
    @. V += $reshape(es, 1, $length(ks)) * modes    
end


function ∂iϕ∂jϕ(ks:: AbstractVector{Float64}, 
                bubbles:: AbstractVector{Bubble}, 
                ball_space:: BallSpace,
                plan:: CFTPlan,
                _x̂_ix̂_j:: x̂_ix̂_j;
                ΔV:: Float64 = 1.):: Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    domes = intersection_domes(bubbles, ball_space)
    @inbounds for (bubble_index, _domes) in domes
        bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                    plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

function Λ∂iϕ∂jϕ(ks:: AbstractVector{Float64}, 
                 bubbles:: AbstractVector{Bubble}, 
                 ball_space:: BallSpace,
                 plan:: CFTPlan,
                 _Λx̂x̂:: Λx̂x̂;
                 ΔV:: Float64 = 1.):: Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    domes = intersection_domes(bubbles, ball_space)
    @inbounds for (bubble_index, _domes) in domes
        bubble_Λ∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                     plan, _Λx̂x̂; ΔV=ΔV)
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

function Λ(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
    return (1. / 2) * ((T1[1] - T1[4])' * (T2[1] - T2[4])) + 2 * (T1[2]' * T2[2])
end

function Λ(T:: AbstractVector{ComplexF64}):: ComplexF64
    return Λ(T, T)
end

export Directional_Π

# Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
function Directional_Π(_n̂:: Vec3, t:: Float64, ωs:: AbstractVector{Float64}, 
                       snapshot:: BubblesSnapShot, ball_space:: BallSpace, plan:: CFTPlan, 
                       _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1.):: Vector{ComplexF64}
    _snap = align_ẑ(_n̂) * snapshot
    bubbles = current_bubbles(_snap, t)
    T = ∂iϕ∂jϕ(ωs, bubbles, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV)
    return @. Λ($eachcol(T)) / $volume(ball_space)
end

# Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
function Directional_Π(_n̂:: Vec3, t:: Float64, ωs:: AbstractVector{Float64}, 
                       snapshot:: BubblesSnapShot, ball_space:: BallSpace, plan:: CFTPlan, 
                       _Λx̂x̂:: Λx̂x̂; ΔV:: Float64 = 1.):: Vector{ComplexF64}
    _snap = align_ẑ(_n̂) * snapshot
    bubbles = current_bubbles(_snap, t)
    ΛT = Λ∂iϕ∂jϕ(ωs, bubbles, ball_space, plan, _Λx̂x̂; ΔV=ΔV)
    return map(eachcol(ΛT)) do v
        2 * v' * v / volume(ball_space)
    end
end

# Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
function Directional_Π(_n̂:: Vec3, t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, 
                       snapshot:: BubblesSnapShot, ball_space:: BallSpace, plan:: CFTPlan, 
                       _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1.):: Vector{ComplexF64}
    t1 ≈ t2 && (return Directional_Π(_n̂, t1, ωs, snapshot, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV))
    _snap = align_ẑ(_n̂) * snapshot
    bubbles1 = current_bubbles(_snap, t1)
    bubbles2 = current_bubbles(_snap, t2)
    T1 = ∂iϕ∂jϕ(ωs, bubbles1, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV)
    T2 = ∂iϕ∂jϕ(ωs, bubbles2, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV)
    return @. Λ($eachcol(T1), $eachcol(T2)) / $volume(ball_space)
end

# Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
function Directional_Π(_n̂:: Vec3, t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, 
                       snapshot:: BubblesSnapShot, ball_space:: BallSpace, plan:: CFTPlan, 
                       _Λx̂x̂:: Λx̂x̂; ΔV:: Float64 = 1.):: Vector{ComplexF64}
    t1 ≈ t2 && (return Directional_Π(_n̂, t1, ωs, snapshot, ball_space, plan, _x̂_ix̂_j; ΔV=ΔV))
    _snap = align_ẑ(_n̂) * snapshot
    bubbles1 = current_bubbles(_snap, t1)
    bubbles2 = current_bubbles(_snap, t2)
    ΛT1 = Λ∂iϕ∂jϕ(ωs, bubbles1, ball_space, plan, _Λx̂x̂; ΔV=ΔV)
    ΛT2 = Λ∂iϕ∂jϕ(ωs, bubbles2, ball_space, plan, _Λx̂x̂; ΔV=ΔV)
    return map(zip(eachcol(ΛT1), eachcol(ΛT2))) do (v1, v2)
        2 * v1' * v2 / volume(ball_space)
    end
end


"""
    Π(t1::Float64, 
      t2::Float64, 
      ωs::AbstractVector{Float64}, 
      snapshot::BubblesSnapShot, 
      ball_space::BallSpace, 
      cft_plan::CFTPlan, 
      angular_integration_plan:: SHPlan,
      _x̂_ix̂_j::x̂_ix̂_j; 
      ΔV::Float64 = 1.)

Computes the isotropic anisotropic stress correlation by integrating 
Directional_Π over all angles using the provided strategy `plan`.

Returns:
- Vector{ComplexF64}: The angle-averaged value (Total Integral / 4π).
"""
function Π(t1::Float64, 
           t2::Float64, 
           ωs::AbstractVector{Float64}, 
           snapshot::BubblesSnapShot, 
           ball_space::BallSpace, 
           cft_plan::CFTPlan, 
           angular_integration_plan:: AbstractAngularIntegrationPlan,
           _x̂_ix̂_j::x̂_ix̂_j; 
           ΔV::Float64 = 1.)
           
    # Wrapper to convert (ϕ, θ) into the 3D direction vector n̂ expected by Directional_Π
    # and ensure the output format matches what SphericalHarmonics expects.
    function f_wrapper(ϕ::Float64, θ::Float64)
        # Convert angles to unit vector [cite: 8, 104]
        n_vec = n̂(SVector{2, Float64}(ϕ, θ)) 
        
        # Call the physics kernel 
        return Directional_Π(n_vec, t1, t2, ωs, snapshot, ball_space, cft_plan, _x̂_ix̂_j; ΔV=ΔV)
    end

    # Return average (integral / 4π) to match previous conventions 
    return integrate_angular(angular_integration_plan, f_wrapper) ./ 4π
end

"""
    Π(t1::Float64, 
      t2::Float64, 
      ωs::AbstractVector{Float64}, 
      snapshot::BubblesSnapShot, 
      ball_space::BallSpace, 
      cft_plan::CFTPlan, 
      angular_integration_plan:: SHPlan,
      _x̂_ix̂_j::x̂_ix̂_j; 
      ΔV::Float64 = 1.)

Computes the isotropic anisotropic stress correlation by integrating 
Directional_Π over all angles using the provided strategy `plan`.

Returns:
- Vector{ComplexF64}: The angle-averaged value (Total Integral / 4π).
"""
function Π(t1::Float64, 
           t2::Float64, 
           ωs::AbstractVector{Float64}, 
           snapshot::BubblesSnapShot, 
           ball_space::BallSpace, 
           cft_plan::CFTPlan, 
           angular_integration_plan:: AbstractAngularIntegrationPlan,
           _Λx̂x̂:: Λx̂x̂; 
           ΔV::Float64 = 1.)
           
    # Wrapper to convert (ϕ, θ) into the 3D direction vector n̂ expected by Directional_Π
    # and ensure the output format matches what SphericalHarmonics expects.
    function f_wrapper(ϕ::Float64, θ::Float64)
        # Convert angles to unit vector [cite: 8, 104]
        n_vec = n̂(SVector{2, Float64}(ϕ, θ)) 
        
        # Call the physics kernel 
        return Directional_Π(n_vec, t1, t2, ωs, snapshot, ball_space, cft_plan, _Λx̂x̂; ΔV=ΔV)
    end

    # Return average (integral / 4π) to match previous conventions 
    return integrate_angular(angular_integration_plan, f_wrapper) ./ 4π
end


end