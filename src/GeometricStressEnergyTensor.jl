module GeometricStressEnergyTensor

import EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using StaticArrays
using LinearAlgebra
using Intervals
import Intervals: IntervalSet
using Rotations
import Base: *, ∈, isempty, ~, ∩
using QuadGK
using DoubleExponentialFormulas

function intersecting(bubble1:: Bubble, bubble2:: Bubble):: Bool
    return euc(bubble1.center, bubble2.center) < bubble1.radius + bubble2.radius
end

struct IntersectionArc
    h:: Float64
    n̂:: Vec3
    dome_like:: Bool
end

function IntersectionArc(n:: Vec3, dome_like:: Bool)
    h = norm(n)
    return IntersectionArc(h, n / h, dome_like)
end

export IntersectionArc

#=
Credit to WolFram Malthworld sphere-sphere intersection article
=#
function λ(r1:: Float64, r2:: Float64, d:: Float64) 
    return (d^2 + r1^2 - r2^2) / 2d
end

function ∩(bubble1:: Bubble, bubble2:: Bubble):: Tuple{IntersectionArc, IntersectionArc}
    n = bubble2.center - bubble1.center
    d = norm(n)
    n̂ = n / d 
    _λ = λ(bubble1.radius, bubble2.radius, d)
    n1 = _λ * n̂
    n2 = -n + n1
    domelike1 = _λ > 0.
    domelike2 = λ(bubble2.radius, bubble1.radius, d) > 0.
    return (IntersectionArc(n1, domelike1), IntersectionArc(n2, domelike2))
end

function intersection_arcs(bubbles:: Bubbles):: Dict{Int, Vector{IntersectionArc}}
    d = Dict{Int, Vector{IntersectionArc}}()
    for i in eachindex(bubbles.bubbles)
        d[i] = Vector{IntersectionArc}()
    end
    for (i, bubble1) in enumerate(bubbles.bubbles)
        for (j̃, bubble2) in enumerate(bubbles.bubbles[(i + 1):end])
            j = j̃ + i
            if intersecting(bubble1, bubble2)
                intersection1, intersection2 = bubble1 ∩ bubble2
                for (k, intersection) in ((i, intersection1), (j, intersection2))
                    push!(d[k], intersection)
                end
            else
                continue
            end        
        end
    end
    return d
end

export intersection_arcs

mod2π(ϕ:: Float64) = mod(ϕ, 2π)

atan2π = mod2π ∘ atan

const NullVec:: Vec3 = Vec3(zeros(3))

const ẑ:: Vec3 = Vec3(0., 0., 1.)

∥(u:: Vec3, v:: Vec3):: Bool = u×v ≈ NullVec

struct PeriodicInterval
    ϕ1:: Float64
    Δ:: Float64
end

∈(ϕ:: Float64, p:: PeriodicInterval):: Bool = mod2π(ϕ - p.ϕ1) <= p.Δ
approxempty(p:: PeriodicInterval):: Bool = p.Δ ≈ 0.
approxentire(p:: PeriodicInterval):: Bool = p.Δ ≈ 2π 

const EmptyArc:: PeriodicInterval = PeriodicInterval(0., 0.)
const FullCircle:: PeriodicInterval = PeriodicInterval(0., 2π)

function complement(p:: PeriodicInterval):: PeriodicInterval
    if approxempty(p)
        return FullCircle
    elseif approxentire(p)
        return EmptyArc
    else 
        return PeriodicInterval(mod2π(p.ϕ1 + p.Δ), 2π - p.Δ)
    end
end

const lib_path = "$(dirname(dirname(pathof(EnvelopeApproximation))))/rust_extensions/benrust/obj/fastrust.so"

rdi(μ, R, n̂, h) = @ccall lib_path.ring_dome_intersect(μ:: Float64, R:: Float64, n̂:: Tuple{Float64, Float64, Float64}, h:: Float64):: Tuple{Float64, Float64}

# This function returns the intersection between the integratoin ring and the dome like region
# of the intersection of 2 bubbles
function ring_dome_intersection(μ′:: Float64, R:: Float64, n̂′:: Vec3, h:: Float64):: PeriodicInterval
    return PeriodicInterval(rdi(μ′, R, n̂′.data, h)...)
end

# A prime indicates that the intersection is in a rotated coordinate system
function Δϕ′(μ′:: Float64, R:: Float64, 
             intersection′:: IntersectionArc):: PeriodicInterval
    n̂′, h = intersection′.n̂, intersection′.h
    _ring_dome_intersection = ring_dome_intersection(μ′, R, n̂′, h) 
    # This function returns the correct arc of integration, in a representation by a single periodic interval.
    intersection′.dome_like && return complement(_ring_dome_intersection)
    return _ring_dome_intersection
end

function IntervalSet(Δϕ:: PeriodicInterval):: IntervalSet{Interval{Float64, Closed, Closed}}
    # Naive use of intervals ignore the fact that the point 0. is ientified with
    # The point 2π, This means we need to fix intervalss that pass through the origin
    # This function assumes Δϕ is smaller than π
    ϕ1 = Δϕ.ϕ1
    ϕ2 = ϕ1 + Δϕ.Δ
    if ϕ2 ≲ 2π
        return IntervalSet(ϕ1 .. ϕ2)
    else
        return IntervalSet([0. .. mod2π(ϕ2), ϕ1 .. 2π])
    end
end

const FullCircleSet:: IntervalSet{Interval{Float64, Closed, Closed}} = IntervalSet(FullCircle)

function Δϕ′(μ′:: Float64, R:: Float64,
             intersection_arcs:: Vector{IntersectionArc}):: IntervalSet
    isempty(intersection_arcs) && return FullCircleSet
    return reduce(∩, (IntervalSet(Δϕ′(μ′, R, intersection_arc)) for intersection_arc in intersection_arcs))
end

function polar_intersection_region(R:: Float64, 
                                   arc:: IntersectionArc):: Tuple{Float64, Float64}
    # This function assumes h < R
    n̂_z = arc.n̂[3]
    n̂_xy = √(1 - n̂_z ^ 2)
    ratio = arc.h / R
    c = n̂_z * ratio
    Δ = n̂_xy * √(1 - ratio ^ 2)
    return (c - Δ, c + Δ)
end

function polar_limits(R:: Float64, arcs:: Vector{IntersectionArc}):: Vector{Float64}
    n = length(arcs)
    regions = Vector{Float64}(undef, 2n)
    for (i, arc) in enumerate(arcs)
        t = polar_intersection_region(R, arc)
        for (j, e) in enumerate(t)
            regions[2i + j - 2] = e
        end
    end
    regions |> unique! |> sort! 
    pushfirst!(regions, -1.)
    return push!(regions, 1.)
end

include("SphericalIntegrands.jl")

struct BubbleArcSurfaceIntegrand <: SphericalIntegrand{MVector{6, Float64}}
    R:: Float64
    arcs:: Vector{IntersectionArc}
end

function ∫_ϕ(basi:: BubbleArcSurfaceIntegrand, μ:: Float64):: MVector{6, Float64}
    V = zeros(MVector{6, Float64})
    intervals = Δϕ′(μ, basi.R, basi.arcs)
    for interval in intervals.items
        V .+= ∫_ϕ(upper_right, μ, interval.first, interval.last)
    end
    return V
end

const ZHat:: SphericalZhat = SphericalZhat()

struct BubbleArcPotentialIntegrand <: SphericalIntegrand{Float64}
    R:: Float64
    arcs:: Vector{IntersectionArc}
end

function ∫_ϕ(bapi:: BubbleArcPotentialIntegrand, μ:: Float64):: Float64
    intervals = Δϕ′(μ, bapi.R, bapi.arcs).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ZHat, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end

# Assume the rotation is right handed, that is of unit determinant (else it would change the dome_like parameter)
function *(rotation:: SMatrix{3, 3, Float64}, arc:: IntersectionArc):: IntersectionArc
     return IntersectionArc(arc.h, rotation * arc.n̂, arc.dome_like)
end

function *(rotation:: SMatrix{3, 3, Float64}, basi:: BubbleArcSurfaceIntegrand):: BubbleArcSurfaceIntegrand
    return BubbleArcSurfaceIntegrand(basi.R, (rotation, ) .* basi.arcs)
end

function fourier_mode(f:: SphericalIntegrand{Float64}, 
                      κ:: Float64; kwargs...):: ComplexF64
    _f(μ:: Float64):: ComplexF64 = cis(-κ * μ) * ∫_ϕ(f, μ)
    return quadgk(_f, -1., 1.; kwargs...)[1]
end

function fourier_mode(f:: SphericalIntegrand{MVector{K, Float64}}, 
                      κ:: Float64; kwargs...):: MVector{K, ComplexF64} where K
    _f(μ:: Float64):: MVector{K, ComplexF64} = cis(-κ * μ) * ∫_ϕ(f, μ)
    return quadgk(_f, -1., 1.; kwargs...)[1]
end

function ∠(k:: Vec3):: Vec3
    (k ∥ ẑ) && return Vec3(0., 0., 0.)
    k_ = norm(k)
    θ = acos(k[3] / k_)
    return ((k / (k_ * sin(θ))) * θ) × ẑ
end

align_ẑ(k:: Vec3):: SMatrix{3, 3, Float64} = SMatrix{3, 3, Float64}(RotationVec(∠(k)...))

export align_ẑ

# The mapping between a 3 x 3 symmetric tensor's double indices and 
#  a vector of length 6
const SymmetricTensorMapping:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2 => (1, 2), 3 => (1, 3), 4 => (2, 2), 5 => (2, 3), 6 => (3, 3))

# The mapping looks like This:
#=
   1       2     3
 #undef    4     5
 #undef  #undef  6
=#


#=
This matrix applies the transformation law:
x̂_ix̂_j = R_li * Rmj * x̂′_l x̂′_m
=#

function symmetric_tensor_inverse_rotation(rotation:: SMatrix{3, 3, Float64}):: SMatrix{6, 6, Float64}
    drot = MMatrix{6, 6, Float64}(undef)
    @inbounds for n ∈ 1:6, k ∈ 1:6
        i, j = SymmetricTensorMapping[k]
        l, m = SymmetricTensorMapping[n]
        if l == m
            drot[k, n] = rotation[l, i] * rotation[m, j] 
        else
            drot[k, n] = (rotation[l, i] * rotation[m, j]) + (rotation[m, i] * rotation[l, j]) 
        end
    end
    return drot
end

export symmetric_tensor_inverse_rotation

function add_bubble_contribution!(V:: MVector{6, ComplexF64}, k:: Vec3, bubble:: Bubble, arcs:: Vector{IntersectionArc},
                                  krotation:: SMatrix{3, 3, Float64}, 
                                  ΔV:: Float64; kwargs...):: MVector{6, ComplexF64}
    mode = fourier_mode(BubbleArcSurfaceIntegrand(bubble.radius, (krotation, ) .* arcs), bubble.radius * norm(k); kwargs...)
    V .+= mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function surface_integral(k:: Vec3, bubbles:: Bubbles, 
                          arcs:: Dict{Int64, Vector{IntersectionArc}},
                          krotation:: SMatrix{3, 3, Float64}, 
                          kdrotation:: SMatrix{6, 6, Float64},
                          ΔV:: Float64; kwargs...):: MVector{6, ComplexF64}
    V = zeros(MVector{6, ComplexF64})
    for (bubble_index, bubble_arcs) in arcs
        add_bubble_contribution!(V, k, bubbles[bubble_index], 
                                 bubble_arcs, krotation, ΔV; kwargs...)
    end
    return kdrotation * V
end

function surface_integral(ks:: Vector{Vec3}, bubbles:: Bubbles;
                          arcs:: Union{Nothing, Dict{Int64, Vector{IntersectionArc}}} = nothing, 
                          krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                          kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing, 
                          ΔV:: Float64 = 1., kwargs...):: Matrix{ComplexF64}
    arcs ≡ nothing && (arcs = intersection_arcs(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    V = Matrix{ComplexF64}(undef, 6, length(ks))
    for ((i, k), krot, kdrot) in zip(enumerate(ks), krotations, kdrotations)
        @views V[:, i] .= surface_integral(k, bubbles, arcs, krot, kdrot, ΔV; kwargs...)
    end
    return permutedims(V)
end

export surface_integral

function bubble_potential_contribution(k:: Vec3, bubble:: Bubble, 
                                       arcs:: Vector{IntersectionArc}, 
                                       krotation:: SMatrix{3, 3, Float64}, 
                                       ΔV:: Float64; kwargs...):: ComplexF64
    mode = fourier_mode(BubbleArcPotentialIntegrand(bubble.radius, (krotation, ) .* arcs), bubble.radius * norm(k); kwargs...)
    return mode * (im * cis(-(k ⋅ bubble.center.coordinates))) * ((-ΔV / norm(k)) *  bubble.radius ^ 2) 
end

function potential_integral(k:: Vec3, bubbles:: Bubbles, 
                            arcs:: Dict{Int64, Vector{IntersectionArc}},
                            krotation:: SMatrix{3, 3, Float64}, 
                            ΔV:: Float64; kwargs...):: ComplexF64
    V = 0.
    for (bubble_index, bubble_arcs) in arcs
        V += bubble_potential_contribution(k, bubbles[bubble_index], 
                                           bubble_arcs, krotation, ΔV; kwargs...)
    end
    return V
end

function potential_integral(ks:: Vector{Vec3}, bubbles:: Bubbles;
                            arcs:: Union{Nothing, Dict{Int64, Vector{IntersectionArc}}} = nothing, 
                            krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                            ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    arcs ≡ nothing && (arcs = intersection_arcs(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return potential_integral.(ks, (bubbles, ), (arcs, ), krotations, (ΔV, ); kwargs...)
end

export potential_integral

const DIAGONAL_INDICES:: Vector{Int} = [1, 4, 6]

function T_ij(ks:: Vector{Vec3}, 
              bubbles:: Bubbles; ΔV:: Float64 = 1., 
              krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
              kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing, 
              kwargs...)
    arcs = intersection_arcs(bubbles)
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    si = surface_integral(ks, bubbles; arcs=arcs, krotations=krotations, 
                          kdrotations=kdrotations, ΔV, kwargs...)
    Ṽ = potential_integral(ks, bubbles; arcs=arcs, krotations=krotations, ΔV=ΔV, kwargs...)
    @views @. si[:, DIAGONAL_INDICES] -= Ṽ
    return si 
end

export T_ij

include("StressEneryTensorContractions.jl")

end