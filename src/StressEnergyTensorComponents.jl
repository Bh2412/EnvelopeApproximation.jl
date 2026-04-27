module StressEnergyTensorComponents

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.EnvelopeAnalysis: ring_domes_complement_intersection!,
    ring_domes_complement_buffers, PeriodicInterval, IntersectionDome,
    intersection_domes, unfold_periodic_bubbles
import EnvelopeApproximation.CFTInterface: CFTPlan, fourier_modes
using EnvelopeApproximation.ChebyshevCFT: VectorChebyshevPlanWithAtol, VectorChebyshevPlan
using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
using StaticArrays

export x̂_ix̂_j, ∂iϕ∂jϕ, dispatch_fourier_modes,
       symmetric_tensor_indices, inverse_symmetric_tensor_indices,
       TwoPointAzimuthalReduction

@inline function ∫_ϕ_x̂_ix̂_j(μ::Float64, p::PeriodicInterval)
    ϕ₁, ϕ₂ = p.ϕ1, p.ϕ1 + p.Δ

    μ² = μ^2
    s² = 1 - μ²
    s  = sqrt(s²)

    Δϕ     = ϕ₂ - ϕ₁
    Δsinϕ  = sin(ϕ₂)  - sin(ϕ₁)
    Δcosϕ  = cos(ϕ₁)  - cos(ϕ₂)
    Δsin2ϕ = sin(2ϕ₂) - sin(2ϕ₁)
    Δcos2ϕ = cos(2ϕ₁) - cos(2ϕ₂)

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
    arcs_buffer::Vector{PeriodicInterval}
    limits_buffer::Vector{Tuple{Float64, Float64}}
    intersection_buffer::Vector{PeriodicInterval}
end

x̂_ix̂_j(n::Int64) = x̂_ix̂_j(ring_domes_complement_buffers(n)...)

function (f::x̂_ix̂_j)(μ::Float64, bubble::Bubble,
                      intersection_domes::Vector{IntersectionDome})::SVector{6, Float64}
    V = SVector{6, Float64}(0., 0., 0., 0., 0., 0.)
    periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes,
                                                             f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V += ∫_ϕ_x̂_ix̂_j(μ, interval)
    end
    return V
end

function dispatch_fourier_modes(f, ks::AbstractVector{Float64}, a::Float64, b::Float64,
                                plan::VectorQuadGKPlan{K})::Matrix{ComplexF64} where {K}
    val, err = fourier_modes(f, ks, a, b, plan)
    return val
end

function dispatch_fourier_modes(f, ks::AbstractVector{Float64}, a::Float64, b::Float64,
                                plan::VectorChebyshevPlan{N, K})::Matrix{ComplexF64} where {N, K}
    return collect(permutedims(fourier_modes(f, ks, a, b, plan)))
end

function dispatch_fourier_modes(f, ks::AbstractVector{Float64}, a::Float64, b::Float64,
                                plan::VectorChebyshevPlanWithAtol{N, K, P})::Matrix{ComplexF64} where {N, K, P}
    return collect(permutedims(fourier_modes(f, ks, a, b, plan)[1]))
end

function bubble_∂iϕ∂jϕ_contribution!(V::AbstractMatrix{ComplexF64},
                                     ks::AbstractVector{Float64},
                                     bubble::Bubble,
                                     domes::Vector{IntersectionDome},
                                     plan::CFTPlan,
                                     _x̂_ix̂_j::x̂_ix̂_j;
                                     ΔV::Float64=1.)
    @assert size(V) == (6, length(ks)) "Output buffer V must be (6, Nk)"
    modes = dispatch_fourier_modes(μ -> _x̂_ix̂_j(μ, bubble, domes), ks * bubble.radius, -1., 1., plan)
    es = map(ks) do k
        cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3.)
    end
    @. V += $reshape(es, 1, $length(ks)) * modes
end

function ∂iϕ∂jϕ(ks::AbstractVector{Float64},
                bubbles::AbstractVector{Bubble},
                space::AbstractSpace,
                boundary_condition::Vacuum,
                plan::CFTPlan,
                _x̂_ix̂_j::x̂_ix̂_j;
                ΔV::Float64=1.)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    domes = intersection_domes(bubbles, space, boundary_condition)
    @inbounds for (bubble_index, _domes) in domes
        bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes,
                                    plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

function ∂iϕ∂jϕ(ks::AbstractVector{Float64},
                bubbles::Bubbles,
                space::BoxSpace,
                boundary_condition::Periodic,
                plan::CFTPlan,
                _x̂_ix̂_j::x̂_ix̂_j;
                ΔV::Float64=1.)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    periodic_bubbles = unfold_periodic_bubbles(bubbles, space)
    bubbles = vcat(bubbles, periodic_bubbles)
    domes = intersection_domes(bubbles, space, Vacuum())
    @inbounds for (bubble_index, _domes) in domes
        bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes,
                                    plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

const symmetric_tensor_indices::Dict{Int, Tuple{Int, Int}} =
    Dict(1 => (1,1), 2 => (1,2), 3 => (1,3), 4 => (2,2), 5 => (2,3), 6 => (3,3))
const inverse_symmetric_tensor_indices::Dict{Tuple{Int, Int}, Int} =
    Dict(zip(values(symmetric_tensor_indices), keys(symmetric_tensor_indices)))

struct TwoPointAzimuthalReduction
    plan::CFTPlan
    buffer::x̂_ix̂_j
end

end
