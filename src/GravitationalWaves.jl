module GravitationalWaves

using EnvelopeApproximation.BubbleBasics: Bubble, Vec3
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor: ring_domes_complement_intersection!, _buffers, PeriodicInterval, polar_limits, IntersectionDome, intersection_domes
import IterTools: partition
import EnvelopeApproximation.ChebyshevCFT: VectorChebyshevPlanWithAtol, chebyshev_coeffs!, scale, translation, fourier_modes
import EnvelopeApproximation.BubblesEvolution: BallSpace
import EnvelopeApproximation.ISWPowerSpectrum: n̂, align_ẑ
using StaticArrays
using HCubature

function ∫_ϕ_x̂_ix̂_j(μ:: Float64, p:: PeriodicInterval):: NTuple{6, Float64}
    ϕ1, ϕ2 = p.ϕ1, p.ϕ1 + p.Δ
    s2 = 1 - μ ^ 2
    s = sqrt(s2)
    return s2 * ((1 / 2) * (ϕ2 - ϕ1) + (1/4) * (sin(2ϕ2) - sin(2ϕ1))), s2 * (1 / 4) * (cos(2ϕ1) - cos(2ϕ2)), (μ * s) * (sin(ϕ2) - sin(ϕ1)), s2 * ((1/2) * (ϕ2 - ϕ1) - (1/4) * (sin(2ϕ2) - sin(2ϕ1))), μ * s * (cos(ϕ1) - cos(ϕ2)), μ ^ 2 * (ϕ2 - ϕ1) 
end

struct x̂_ix̂_j
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

x̂_ix̂_j(n:: Int64) = x̂_ix̂_j(_buffers(n)...)

function (f:: x̂_ix̂_j)(μ:: Float64, bubble:: Bubble, 
                      intersection_domes:: Vector{IntersectionDome}):: MVector{6, Float64}
    V = zeros(MVector{6, Float64})
    periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes, 
                                                             f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V .+= ∫_ϕ_x̂_ix̂_j(μ, interval)
    end
    return V
end

# function bubble_∂iϕ∂jϕ_contribution!(V:: AbstractMatrix{ComplexF64},
#                                      ks:: AbstractVector{Float64}, 
#                                      bubble:: Bubble, 
#                                      domes:: Vector{IntersectionDome}, 
#                                      chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#                                      _x̂_ix̂_j:: x̂_ix̂_j; 
#                                      ΔV:: Float64 = 1.) where N
#     @assert size(V) == (length(ks), 6) "The output vector must be of the same length of the input k vector"
#     _polar_limits = polar_limits(bubble.radius, domes)
#     @inbounds for (μ1, μ2) in partition(_polar_limits, 2, 1)
#         s, t = scale(μ1, μ2), translation(μ1, μ2)
#         chebyshev_coeffs!(μ -> _x̂_ix̂_j(μ, bubble, domes), μ1, μ2, chebyshev_plan)
#         @inbounds for (i, k) in enumerate(ks)
#             e = cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3)
#             @. V[i, :] += e * $fourier_mode(k, chebyshev_plan, s, t) # ∂_iφ∂_jφ contribution
#         end
#     end
# end

function bubble_∂iϕ∂jϕ_contribution!(V:: AbstractMatrix{ComplexF64},
                                     ks:: AbstractVector{Float64}, 
                                     bubble:: Bubble, 
                                     domes:: Vector{IntersectionDome}, 
                                     chebyshev_plan:: VectorChebyshevPlanWithAtol{N, 6, P}, 
                                     _x̂_ix̂_j:: x̂_ix̂_j; 
                                     ΔV:: Float64 = 1.) where {N, P}
    @assert size(V) == (length(ks), 6) "The output vector must be of the same length of the input k vector"
    modes = fourier_modes(μ -> _x̂_ix̂_j(μ, bubble, domes), ks, -1., 1., chebyshev_plan)[1]
    es = map(ks) do k
        cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3)
    end
    @. V += $reshape(es, $length(ks), 1) * modes    
end
    

function ∂iϕ∂jϕ(ks:: AbstractVector{Float64}, 
                bubbles:: AbstractVector{Bubble}, 
                ball_space:: BallSpace,
                chebyshev_plan:: VectorChebyshevPlanWithAtol{N, 6, P},
                _x̂_ix̂_j:: x̂_ix̂_j;
                ΔV:: Float64 = 1.):: Matrix{ComplexF64} where {N, P}
    V = zeros(ComplexF64, length(ks), 6)
    domes = intersection_domes(bubbles, ball_space)
    @inbounds for (bubble_index, _domes) in domes
    bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

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
                       ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlanWithAtol{N, 6, P}, 
                       _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1.):: Vector{ComplexF64} where {N, P}
    _snap = align_ẑ(_n̂) * snapshot
    bubbles1 = current_bubbles(_snap, t1)
    bubbles2 = current_bubbles(_snap, t2)
    T1 = ∂iϕ∂jϕ(ωs, bubbles1, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
    T2 = ∂iϕ∂jϕ(ωs, bubbles2, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
    return @. Λ($eachrow(T1), $eachrow(T2)) / $volume(ball_space)
end

export Π

function Π(t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
           ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlanWithAtol{N, 6, P}, 
           _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., kwargs...):: Tuple{Vector{ComplexF64}, Float64} where {N, P}
    function f(_n̂:: SVector{2, Float64}):: Vector{Float64}
        ϕ, θ = _n̂
        return @. 2 * real($Directional_Π($n̂(ϕ, θ), t1, t2, ωs, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV) * $sin(θ))
    end
    v, err = hcubature(f, SVector(0., 0.,), SVector(2π, π / 2); kwargs...)  # It is enough to integrate over half the ski
    return v ./ 4π, err / 4π
end

end