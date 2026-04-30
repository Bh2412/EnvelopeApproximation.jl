module StressEnergyTensorComponents

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.EnvelopeAnalysis: ring_domes_complement_intersection!,
    ring_domes_complement_buffers, PeriodicInterval, IntersectionDome,
    intersection_domes, append_periodic_bubbles!, original_bubble_groups
import EnvelopeApproximation.CFTInterface: CFTPlan, fourier_modes
using EnvelopeApproximation.ChebyshevCFT: VectorChebyshevPlanWithAtol, VectorChebyshevPlan
using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
using StaticArrays

export x̂_ix̂_j, ∂iϕ∂jϕ, dispatch_fourier_modes,
       symmetric_tensor_indices, inverse_symmetric_tensor_indices,
       TwoPointAzimuthalReduction,
       T_i_j_kernel, bubble_T_i_j_contribution!, T_ij

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

@inline function ∫_ϕ_T_i_j_kernel(μ::Float64, p::PeriodicInterval)
    ϕ₁, ϕ₂ = p.ϕ1, p.ϕ1 + p.Δ

    μ² = μ^2
    s² = 1 - μ²
    s  = sqrt(s²)

    Δϕ     = ϕ₂ - ϕ₁
    Δsinϕ  = sin(ϕ₂)  - sin(ϕ₁)
    Δcosϕ  = cos(ϕ₁)  - cos(ϕ₂)
    Δsin2ϕ = sin(2ϕ₂) - sin(2ϕ₁)
    Δcos2ϕ = cos(2ϕ₁) - cos(2ϕ₂)

    return SVector{7, Float64}(
        s² * (0.5Δϕ + 0.25Δsin2ϕ), # xx
        s² * 0.25Δcos2ϕ,           # xy
        μ*s * Δsinϕ,               # xz
        s² * (0.5Δϕ - 0.25Δsin2ϕ), # yy
        μ*s * Δcosϕ,               # yz
        μ² * Δϕ,                   # zz
        μ  * Δϕ                    # potential kernel
    )
end

mutable struct x̂_ix̂_j
    arcs_buffer::Vector{PeriodicInterval}
    limits_buffer::Vector{Tuple{Float64, Float64}}
    intersection_buffer::Vector{PeriodicInterval}
end

x̂_ix̂_j(n::Int64) = x̂_ix̂_j(ring_domes_complement_buffers(n)...)

mutable struct T_i_j_kernel
    arcs_buffer::Vector{PeriodicInterval}
    limits_buffer::Vector{Tuple{Float64, Float64}}
    intersection_buffer::Vector{PeriodicInterval}
end

T_i_j_kernel(n::Int64) = T_i_j_kernel(ring_domes_complement_buffers(n)...)

function (f::T_i_j_kernel)(μ::Float64, bubble::Bubble,
                           intersection_domes::Vector{IntersectionDome})::SVector{7, Float64}
    V = SVector{7, Float64}(0., 0., 0., 0., 0., 0., 0.)
    periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes,
                                                            f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V += ∫_ϕ_T_i_j_kernel(μ, interval)
    end
    return V
end

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

function bubble_T_i_j_contribution!(V::AbstractMatrix{ComplexF64},
                                    ks::AbstractVector{Float64},
                                    bubble::Bubble,
                                    domes::Vector{IntersectionDome},
                                    plan::CFTPlan,
                                    kernel::T_i_j_kernel;
                                    ΔV::Float64=1.)
    @assert size(V) == (6, length(ks)) "Output buffer V must be (6, Nk)"
    R, z = bubble.radius, bubble.center.coordinates[3]
    modes = dispatch_fourier_modes(μ -> kernel(μ, bubble, domes), ks * R, -1., 1., plan)

    kinetic_es = map(k -> cis(-k * z) * (ΔV * R^3 / 3.), ks)
    @. V += $reshape(kinetic_es, 1, $length(ks)) * $view(modes, 1:6, :)

    @inbounds for (k_idx, k) in enumerate(ks)
        potential_scalar = cis(-k * z) * (im * ΔV * R^2 / k) * modes[7, k_idx]
        V[1, k_idx] += potential_scalar  # xx: δ₁₁ = 1
        V[4, k_idx] += potential_scalar  # yy: δ₂₂ = 1
        V[6, k_idx] += potential_scalar  # zz: δ₃₃ = 1
    end
end

function contribution_indices(n_bubbles::Int, bubble_indices):: Vector{Int}
    bubble_indices isa Colon && return Base.OneTo(n_bubbles)
    if bubble_indices isa Integer
        bubble_indices < 1 && throw(ArgumentError("bubble_indices must contain positive indices"))
        return bubble_indices <= n_bubbles ? (bubble_indices:bubble_indices) : Int[]
    end

    selected = Int[]
    for bubble_index in bubble_indices
        bubble_index < 1 && throw(ArgumentError("bubble_indices must contain positive indices"))
        bubble_index <= n_bubbles && push!(selected, Int(bubble_index))
    end
    return selected
end

function periodic_selected_indices(bubbles::Bubbles, box::BoxSpace, bubble_indices)::  Tuple{Bubbles, Vector{Int}}
    n_original = length(bubbles)
    selected_originals = contribution_indices(n_original, bubble_indices)
    isempty(selected_originals) && return Bubble[], Int[]

    origin_map = Int[]
    periodic_bubbles = append_periodic_bubbles!(collect(bubbles), box, origin_map)
    groups = original_bubble_groups(origin_map, n_original)

    selected_indices = Int[]
    for original_index in selected_originals
        append!(selected_indices, groups[original_index])
    end
    return periodic_bubbles, selected_indices
end

function ∂iϕ∂jϕ(ks::AbstractVector{Float64},
                bubbles::AbstractVector{Bubble},
                space::AbstractSpace,
                boundary_condition::Vacuum,
                plan::CFTPlan,
                _x̂_ix̂_j::x̂_ix̂_j;
                ΔV::Float64=1.,
                bubble_indices=:)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    selected_indices = contribution_indices(length(bubbles), bubble_indices)
    isempty(selected_indices) && return V

    domes = intersection_domes(bubbles, space, boundary_condition)
    @inbounds for bubble_index in selected_indices
        _domes = domes[bubble_index]
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
                ΔV::Float64=1.,
                bubble_indices=:)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    bubbles, selected_indices = periodic_selected_indices(bubbles, space, bubble_indices)
    isempty(selected_indices) && return V

    domes = intersection_domes(bubbles, space, Vacuum())

    @inbounds for bubble_index in selected_indices
        _domes = domes[bubble_index]
        bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], _domes,
                                    plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

function T_ij(ks::AbstractVector{Float64},
              bubbles::AbstractVector{Bubble},
              space::AbstractSpace,
              boundary_condition::Vacuum,
              plan::CFTPlan,
              kernel::T_i_j_kernel;
              ΔV::Float64=1.,
              bubble_indices=:)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    selected_indices = contribution_indices(length(bubbles), bubble_indices)
    isempty(selected_indices) && return V

    domes = intersection_domes(bubbles, space, boundary_condition)
    @inbounds for bubble_index in selected_indices
        _domes = domes[bubble_index]
        bubble_T_i_j_contribution!(V, ks, bubbles[bubble_index], _domes,
                                   plan, kernel; ΔV=ΔV)
    end
    return V
end

function T_ij(ks::AbstractVector{Float64},
              bubbles::Bubbles,
              space::BoxSpace,
              boundary_condition::Periodic,
              plan::CFTPlan,
              kernel::T_i_j_kernel;
              ΔV::Float64=1.,
              bubble_indices=:)::Matrix{ComplexF64}
    V = zeros(ComplexF64, 6, length(ks))
    bubbles, selected_indices = periodic_selected_indices(bubbles, space, bubble_indices)
    isempty(selected_indices) && return V

    domes = intersection_domes(bubbles, space, Vacuum())

    @inbounds for bubble_index in selected_indices
        _domes = domes[bubble_index]
        bubble_T_i_j_contribution!(V, ks, bubbles[bubble_index], _domes,
                                   plan, kernel; ΔV=ΔV)
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
