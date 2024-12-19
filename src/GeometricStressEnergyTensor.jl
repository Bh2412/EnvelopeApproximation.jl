module GeometricStressEnergyTensor

import EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using StaticArrays
using LinearAlgebra
using Intervals
import Intervals: IntervalSet
using Rotations
import Base: *, ∈, isempty, ~, ∩, convert
using QuadGK
using IterTools
using DoubleExponentialFormulas

include("GeometricStressEnergyTensor/IntersectionDome.jl")
include("GeometricStressEnergyTensor/PeriodicInterval.jl")
include("GeometricStressEnergyTensor/RingDomeIntersection.jl")
include("GeometricStressEnergyTensor/PeriodicIntersection.jl")

function ring_domes_intersection!(μ′:: Float64, R:: Float64, intersection_domes:: Vector{IntersectionDome}, 
                                  arcs_buffer:: Vector{PeriodicInterval}, 
                                  limits_buffer:: Vector{Tuple{Float64, Float64}},
                                  intersection_buffer:: Vector{PeriodicInterval}):: AbstractVector{PeriodicInterval}
    isempty(intersection_domes) && return @views intersection_buffer[1:0]
    for (i, arc) in enumerate(intersection_domes)
        p = ring_dome_intersection(μ′, R, arc)
        approxempty(p) && return @views intersection_buffer[1:0]
        arcs_buffer[i] = p
    end
    return periodic_intersection!(arcs_buffer, limits_buffer, intersection_buffer)
end

function polar_intersection_region(R:: Float64, 
                                   dome:: IntersectionDome):: Tuple{Float64, Float64}
    # This function assumes h < R
    n̂_z = dome.n̂[3]
    n̂_xy = √(1 - n̂_z ^ 2)
    ratio = dome.h / R
    c = n̂_z * ratio
    Δ = n̂_xy * √(1 - ratio ^ 2)
    return (c - Δ, c + Δ)
end

function polar_limits(R:: Float64, domes:: Vector{IntersectionDome}):: Vector{Float64}    
    regions = (x for dome in domes for t in polar_intersection_region(R, dome) for x in t if abs(x) <= 1.)
    regions = regions |> collect |> unique! |> sort!
    pushfirst!(regions, -1.)
    return push!(regions, 1.)
end

include("GeometricStressEnergyTensor/BasicSphericalIntegrands.jl")
include("GeometricStressEnergyTensor/CommonSphericalIntegrand.jl")
include("GeometricStressEnergyTensor/Rotations.jl")
include("GeometricStressEnergyTensor/FourierModes.jl")

function add_bubble_contribution!(V:: MVector{6, ComplexF64}, k:: Vec3, bubble:: Bubble, domes:: Vector{IntersectionDome},
                                  krotation:: SMatrix{3, 3, Float64}, 
                                  ΔV:: Float64; kwargs...):: MVector{6, ComplexF64}
    mode = fourier_mode(BubbleArcSurfaceIntegrand(bubble.radius, (krotation, ) .* domes), bubble.radius * norm(k); kwargs...)
    V .+= mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function surface_integral(k:: Vec3, bubbles:: Bubbles, 
                          domes:: Dict{Int64, Vector{IntersectionDome}},
                          krotation:: SMatrix{3, 3, Float64}, 
                          kdrotation:: SMatrix{6, 6, Float64},
                          ΔV:: Float64; kwargs...):: MVector{6, ComplexF64}
    V = zeros(MVector{6, ComplexF64})
    for (bubble_index, bubble_arcs) in domes
        add_bubble_contribution!(V, k, bubbles[bubble_index], 
                                 bubble_arcs, krotation, ΔV; kwargs...)
    end
    return kdrotation * V
end

function surface_integral(ks:: Vector{Vec3}, bubbles:: Bubbles;
                          domes:: Union{Nothing, Dict{Int64, Vector{IntersectionDome}}} = nothing, 
                          krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                          kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing, 
                          ΔV:: Float64 = 1., kwargs...):: Matrix{ComplexF64}
    domes ≡ nothing && (domes = intersection_domes(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    V = Matrix{ComplexF64}(undef, 6, length(ks))
    for ((i, k), krot, kdrot) in zip(enumerate(ks), krotations, kdrotations)
        @views V[:, i] .= surface_integral(k, bubbles, domes, krot, kdrot, ΔV; kwargs...)
    end
    return permutedims(V)
end

export surface_integral

function bubble_potential_contribution(k:: Vec3, bubble:: Bubble, 
                                       domes:: Vector{IntersectionDome}, 
                                       krotation:: SMatrix{3, 3, Float64}, 
                                       ΔV:: Float64; kwargs...):: ComplexF64
    mode = fourier_mode(BubbleArcPotentialIntegrand(bubble.radius, (krotation, ) .* domes), bubble.radius * norm(k); kwargs...)
    return mode * (im * cis(-(k ⋅ bubble.center.coordinates))) * ((-ΔV / norm(k)) *  bubble.radius ^ 2) 
end

function potential_integral(k:: Vec3, bubbles:: Bubbles, 
                            domes:: Dict{Int64, Vector{IntersectionDome}},
                            krotation:: SMatrix{3, 3, Float64}, 
                            ΔV:: Float64; kwargs...):: ComplexF64
    V = 0.
    for (bubble_index, bubble_arcs) in domes
        V += bubble_potential_contribution(k, bubbles[bubble_index], 
                                           bubble_arcs, krotation, ΔV; kwargs...)
    end
    return V
end

function potential_integral(ks:: Vector{Vec3}, bubbles:: Bubbles;
                            domes:: Union{Nothing, Dict{Int64, Vector{IntersectionDome}}} = nothing, 
                            krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                            ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    domes ≡ nothing && (domes = intersection_domes(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return potential_integral.(ks, (bubbles, ), (domes, ), krotations, (ΔV, ); kwargs...)
end

export potential_integral

const DIAGONAL_INDICES:: Vector{Int} = [1, 4, 6]

function T_ij(ks:: Vector{Vec3}, 
              bubbles:: Bubbles; ΔV:: Float64 = 1., 
              krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
              kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing, 
              kwargs...)
    domes = intersection_domes(bubbles)
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    si = surface_integral(ks, bubbles; domes=domes, krotations=krotations, 
                          kdrotations=kdrotations, ΔV, kwargs...)
    Ṽ = potential_integral(ks, bubbles; domes=domes, krotations=krotations, ΔV=ΔV, kwargs...)
    @views @. si[:, DIAGONAL_INDICES] -= Ṽ
    return si 
end

export T_ij

function bubble_k̂ik̂j∂iφ∂jφ_contribution(k:: Vec3, bubble:: Bubble, 
    domes:: Vector{IntersectionDome}, 
    krotation:: SMatrix{3, 3, Float64}, 
    ΔV:: Float64; kwargs...):: ComplexF64
# Rotate to a coordinate system in which k̂ik̂j = δi3δj3
mode = fourier_mode(BubbleArck̂ik̂j∂iφ∂jφ(bubble.radius, (krotation, ) .* domes), bubble.radius * norm(k); kwargs...)
return mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function k̂ik̂jTij(k:: Vec3, bubbles:: Bubbles, 
domes:: Dict{Int64, Vector{IntersectionDome}},
krotation:: SMatrix{3, 3, Float64}, 
ΔV:: Float64; kwargs...):: ComplexF64
V = 0.
for (bubble_index, bubble_arcs) in domes
V += bubble_k̂ik̂j∂iφ∂jφ_contribution(k, bubbles[bubble_index], 
        bubble_arcs, krotation, ΔV; kwargs...)
V -= bubble_potential_contribution(k, bubbles[bubble_index], 
       bubble_arcs, krotation, ΔV; kwargs...)
end
return V
end

function k̂ik̂jTij(ks:: Vector{Vec3}, bubbles:: Bubbles;
                 domes:: Union{Nothing, Dict{Int64, Vector{IntersectionDome}}} = nothing, 
                 krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                 ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    domes ≡ nothing && (domes = intersection_domes(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return @. k̂ik̂jTij(ks, (bubbles, ), (domes, ), krotations, (ΔV, ); kwargs...)
end

export k̂ik̂jTij

function bubble_Ŋ_contribution(k:: Vec3, bubble:: Bubble, 
    domes:: Vector{IntersectionDome}, 
    krotation:: SMatrix{3, 3, Float64}, 
    ΔV:: Float64; kwargs...):: ComplexF64
    mode = fourier_mode(BubbleArcŊ(bubble.radius, (krotation, ) .* domes), bubble.radius * norm(k); kwargs...)
    return mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function ŋ_source(k:: Vec3, bubbles:: Bubbles, 
                  domes:: Dict{Int64, Vector{IntersectionDome}},
                  krotation:: SMatrix{3, 3, Float64}, 
                  ΔV:: Float64; kwargs...):: ComplexF64
    V = 0.
    for (bubble_index, bubble_arcs) in domes
        V += bubble_Ŋ_contribution(k, bubbles[bubble_index], 
            bubble_arcs, krotation, ΔV; kwargs...)
    end
    return V
end

function ŋ_source(ks:: Vector{Vec3}, bubbles:: Bubbles;
                  domes:: Union{Nothing, Dict{Int64, Vector{IntersectionDome}}} = nothing, 
                  krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                  ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    domes ≡ nothing && (domes = intersection_domes(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return ŋ_source.(ks, (bubbles, ), (domes, ), krotations, (ΔV, ); kwargs...)
end

export ŋ_source


end