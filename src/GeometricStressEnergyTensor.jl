module GeometricStressEnergyTensor

import EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.ChebyshevCFT
import EnvelopeApproximation.ChebyshevCFT: fourier_mode, scale, translation
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
    isempty(intersection_domes) && begin 
        intersection_buffer[1] = FullCircle
        return @views intersection_buffer[1:1]
    end
    length(intersection_domes) == 1 && begin
        intersection_buffer[1] = ring_dome_intersection(μ′, R, intersection_domes[1])
        return @views intersection_buffer[1:1]
    end
    i = 1
    for dome in intersection_domes
        p = ring_dome_intersection(μ′, R, dome)
        approxempty(p) && return @views intersection_buffer[1:0]
        approxentire(p) && continue
        arcs_buffer[i] = p
        i += 1
    end
    i == 1 && return @views arcs_buffer[1:0]
    i == 2 && return @views arcs_buffer[1:1]
    return @views periodic_intersection!(arcs_buffer[1:(i - 1)], limits_buffer, intersection_buffer)
end

struct Δ
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

Δ(n:: Int64) = Δ(_buffers(n)...)

function (δ:: Δ)(μ:: Float64, bubble:: Bubble, 
                 intersection_domes:: Vector{IntersectionDome}):: Float64
                 periodic_intervals = ring_domes_intersection!(μ, bubble.radius, intersection_domes, 
                                                               δ.arcs_buffer, δ.limits_buffer, δ.intersection_buffer)
    return sum((p.Δ for p in periodic_intervals), init=0.)
end

export Δ

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

function k̂ik̂jTij(ks:: AbstractVector{Float64}, 
                 bubbles:: AbstractVector{Bubble}, 
                 chebyshev_plan:: First3MomentsChebyshevPlan{N},
                 _Δ:: Δ;
                 ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    V = zeros(ComplexF64, length(ks))
    domes = intersection_domes(bubbles)
    @inbounds for (bubble_index, _domes) in domes
    bubble_k̂ik̂jTij_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                 chebyshev_plan, _Δ; ΔV=ΔV)
    end
    return V
end

export k̂ik̂jTij

function k̂ik̂j∂_iφ∂_jφ(ks:: AbstractVector{Float64}, 
                      bubbles:: AbstractVector{Bubble}, 
                      chebyshev_plan:: First3MomentsChebyshevPlan{N},
                      _Δ:: Δ;
                      ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    V = zeros(ComplexF64, length(ks))
    domes = intersection_domes(bubbles)
    @inbounds for (bubble_index, _domes) in domes
        bubble_k̂ik̂j∂_iφ∂_jφ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                          chebyshev_plan, _Δ; ΔV=ΔV)
    end
    return V
end

export k̂ik̂j∂_iφ∂_jφ

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

function bubble_k̂ik̂jTij_contribution!(V:: AbstractVector{ComplexF64},
                                      ks:: AbstractVector{Float64}, 
                                      bubble:: Bubble, 
                                      domes:: Vector{IntersectionDome}, 
                                      chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                                      _Δ:: Δ; 
                                      ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert length(V) == length(ks) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    @inbounds for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _Δ(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        @inbounds for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3])
            _, c1, c2 = fourier_mode(k * bubble.radius, chebyshev_plan, s, t)
            V[i] += c2 * (ΔV * (bubble.radius ^ 3) / 3) * e # ∂_iφ∂_jφ contribution
            V[i] -= c1 * (-im * ΔV) * e / k * (bubble.radius ^ 2) # potential contribution
        end
    end
    return V
end

function bubble_k̂ik̂j∂_iφ∂_jφ_contribution!(V:: AbstractVector{ComplexF64},
                                           ks:: AbstractVector{Float64}, 
                                           bubble:: Bubble, 
                                           domes:: Vector{IntersectionDome}, 
                                           chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                                           _Δ:: Δ; 
                                           ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert length(V) == length(ks) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    @inbounds for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _Δ(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        @inbounds for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3])
            _, _, c2 = fourier_mode(k * bubble.radius, chebyshev_plan, s, t)
            V[i] += c2 * (ΔV * (bubble.radius ^ 3) / 3) * e # ∂_iφ∂_jφ contribution
        end
    end
    return V
end

function bubble_Ŋ_contribution!(V:: AbstractVector{ComplexF64},
                                ks:: AbstractVector{Float64}, 
                                bubble:: Bubble, 
                                domes:: Vector{IntersectionDome}, 
                                chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                                _Δ:: Δ; 
                                ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert length(V) == length(ks) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _Δ(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3])
            c0, _, c2 = fourier_mode(k * bubble.radius, chebyshev_plan, s, t)
        V[i] += (c2 - (1. / 3) * c0) * (ΔV * (bubble.radius ^ 3) / 3) * e # ∂_iφ∂_jφ contribution
        end
    end
    return V
end

end