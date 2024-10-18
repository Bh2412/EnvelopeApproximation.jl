module GeometricStressEnergyTensor

using EnvelopeApproximation.BubbleBasics
using LinearAlgebra
using Intervals
using Rotations

function intersecting(bubble1:: Bubble, bubble2:: Bubble):: Bool
    return euc(bubble1.center - bubble2.center) < bubble1.radius + bubble2.radius
end

struct Intersection
    h:: Float64
    n̂:: Vec3
    dome_like:: Bool
end

function Intersection(n:: Vec3, includes_center:: Bool)
    h = norm(n)
    return Intersection(h, n / h, includes_center)
end

#=
Credit to WolFram Malthworld sphere-sphere intersection article
=#
function λ(r1:: Float64, r2:: Float64, d:: Float64) 
    x = (d^2 + r1^2 - r2^2) / 2d
    return x
end

function ∩(bubble1:: Bubble, bubble2:: Bubble):: Tuple{Intersection, Interection}
    n = bubble2.center - bubble1.center
    d = norm(n)
    n̂ = n / d 
    _λ = λ(bubble1.radius, bubble2.radius, d)
    n1 = _λ * n̂
    n2 = -n + n1
    in1 = sign(d^2 + bubble1.radius ^ 2 - bubble2.radius ^ 2)
    in2 = sign(d^2 + bubble2.radius ^ 2 - bubble1.radius ^ 2)
    return (Intersection(n1, in1), Intersection(n2, in2))
end

function intersections(bubbles:: Bubbles):: Dict{Int, Vector{Intersection}}
    d = Dict{Int, Vector{Intersection}}()
    for (i, bubble1) in enumerate(bubbles)
        for (j̃, bubble2) in bubbles[(i + 1):end]
            j = j̃ + i
            if intersecting(bubble1, bubble2)
                intersection1, intersection2 = bubble1 ∩ bubble2
                for (k, intersection) in ((i, intersection1), (j, intersection2))
                    if k ∈ keys(d)
                        push!(d[k], intersection)
                    else
                        d[k] = Vector{Intersection}([intersection])
                    end
                end
            else
                continue
            end        
        end
    end
    return d
end

mod2π(ϕ:: Float64) = mod(ϕ, 2π)

atan2π = mod2π ∘ atan

const NullVec:: Vec3 = Vec3(zeros(3))

const ẑ:: Vec3 = Vec3(0., 0., 1.)

∥(u:: Vec3, v:: Vec3):: Bool = u×v ≈ NullVec

const EmptyInterval:: Interval{Float64, Closed, Closed} = 2π .. 0.
const EntireRing:: Interval{Float64, Closed, Closed} = 0. .. 2π

function ∠(k:: Vec3):: Vec3
    ∥(k) && return Vec3(0., 0., 0.)
    k_ = norm(k)
    θ = acos(k[3] / k_)
    return ẑ × (k / k_ * sin(θ)) * θ
end

align_ẑ(k:: Vec3):: SMatrix{3, 3, Float64} = SMatrix{3, 3, Float64}(RotationVec(∠(k...)))

function Δϕ′(μ′:: Float64, R:: Float64, n̂′:: Vec3, h:: Float64):: Interval{Float64, Closed, Closed}
    # This function assumes n̂′ is not parallel to the sphere of the integration ring
    s′ = √(1 - μ′ ^ 2)
    d, sgn = begin
        x = (h - μ′ * R) / √(n̂′[1] ^ 2 + n̂′[2] ^ 2)
        abs(x), sign(x)
    end
    if abs(d) >= R * s′
        return EmptyInterval
    end
    α = atan2π(n̂′[2] * sgn, n̂′[1] * sgn)
    Δ = acos(d / (R * s′))
    # Returns the interval that describes the Dome!!! of the intersection, that is the short arc of the intersection
    return α - Δ .. α + Δ
end

function apply_periodicity(Δϕ:: Interval{Float64, Closed, Closed}):: IntervalSet{Interval{Float64, Closed, Closed}}
    if Δϕ.right >= Δϕ.left
        return IntervalSet([Δϕ])
    else
        return IntervalSet([0. .. Δϕ.right, Δϕ.left .. 2π])
    end
end

function Δϕ′(μ′:: Float64, R:: Float64, krotation:: SMatrix{3, 3, Float64},
             intersection:: Intersection):: Interval{Float64, Closed, Closed}
    n̂′ = krotation * intersection.n̂
    if n̂′ ∥ ẑ
        if (μ′ * R * sign(n̂′[3])) >= intersection.h
            return EntireRing
        else
            return EmptyInterval
        end
    else
        _Δϕ′ = Δϕ′(μ′, R, n̂′, intersection.h)        
    end
    # This function returns the correctt arc of the intersection, in a representation by a single interval.
    if intersection.dome_like
        return _Δϕ′
    else
        return _Δϕ′.left .. _Δϕ′.right
    end
end

function Δϕ′(μ′:: Float64, R:: Float64, krotation:: SMatrix{3, 3, Float64}, 
             intersections:: Vector{Intersection}):: Vector{Interval{Float64, Closed, Closed}}
    _Δϕ′(intersection:: Intersectoin):: Interval{Float64, Closed, Closed} = Δϕ′(μ′, R, krotation, intersection)
    return @. $reduce(∩, apply_periodicity(_Δϕ′(intersections)))
end

abstract type SphericalIntegrand{T} end

function (si:: SphericalIntegrand{T})(μ:: Float64, ϕ:: Float64):: T where T  
    throw(error("Not Implemented"))
end

function (si:: SphericalIntegrand{T})(μϕ:: Tuple{Float64, Float64}):: T where T
    return si(μϕ...)
end

function ∫_ϕ(si:: SphericalIntegrand{T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: T
    throw(error("Not Implemented"))
end

struct SphericalMultiplicationIntegrand{T} <: SphericalIntegrand{T}
    components:: NTuple{K, SphericalIntegrand} where K
end

function (m:: SphericalMultiplicationIntegrand{T})(μ:: Float64, ϕ:: Float64):: T where T
    prod(c(μ, ϕ) for c in m.components)
end

function *(si1:: SphericalIntegrand{Float64}, si2:: SphericalIntegrand{SVector{K, Float64}}):: SphericalMultiplicationIntegrand{SVector{K, Float64}} where K 
    SphericalMultiplicationIntegrand{SVector{Float64}}((si1, si2))
end

struct SphericalDirectSumIntegrand{K, T} <: SphericalIntegrand{Vector{T}}
    components:: NTuple{K, Z} where Z <: SphericalIntegrand{T}
end

function (ds:: SphericalDirectSumIntegrand{K, T})(μ:: Float64, ϕ:: Float64):: Vector{T}
    invoke.(ds.components, Tuple{Float64, Float64}, μ, ϕ)
end

function (ds:: SphericalDirectSumIntegrand{K, T})(V:: Vector{T}, μ:: Float64, ϕ:: Float64):: Vector{T}
    @. V = invoke(ds.components, Tuple{Float64, Float64}, μ, ϕ)
    return V
end

function ⊕(si1:: SphericalIntegrand{T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{2, T} where T
    SphericalDirectSumIntegrand{2, T}((si1, si2))
end

function ⊕(si1:: SphericalDirectSumIntegrand{K, T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{K + 1, T} where {K, T}
    SphericalDirectSumIntegrand{K+1, T}(((si1.components..., si2)))
end

function ∫_ϕ(sdsi:: SphericalDirectSumIntegrand{K, T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Vector{T}
    return ∫_ϕ.(sdsi.components, μ, ϕ1, ϕ2)
end

function ∫_ϕ!(V:: Vector{T}, sdsi:: SphericalDirectSumIntegrand{K, T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Vector{T}
    return @. V = ∫_ϕ(sdsi.components, (μ, ), (ϕ1, ), (ϕ2, ))
end

abstract type TensorDirection <: SphericalIntegrand{Float64} end
struct SphericalTrace <: TensorDirection end
struct SphericalXhat <: TensorDirection end
struct SphericalYhat <: TensorDirection end
struct SphericalZhat <: TensorDirection end
struct SphericalXX <: TensorDirection end
struct SphericalXY <: TensorDirection end
SphericalYX = SphericalXY
struct SphericalXZ <: TensorDirection end
SphericalZX = SphericalXZ
struct SphericalYY <: TensorDirection end
struct SphericalYZ <: TensorDirection end
struct SphericalZZ <: TensorDirection end

(st:: SphericalTrace)(μ:: Float64, ϕ:: Float64):: Float64 = 1.
∫_ϕ(st:: SphericalTrace, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = ϕ2 - ϕ1
(st:: SphericalXhat)(μ:: Float64, ϕ:: Float64):: Float64 = √(1 - μ^2) * cos(ϕ)
∫_ϕ(st:: SphericalXhat, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = √(1 - μ ^ 2) * (sin(ϕ2) - sin(ϕ1))
(st:: SphericalYhat)(μ:: Float64, ϕ:: Float64):: Float64 = √(1 - μ^2) * sin(ϕ)
∫_ϕ(st:: SphericalYhat, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = √(1 - μ ^ 2) * (cos(ϕ1) - cos(ϕ2))
(st:: SphericalZhat)(μ:: Float64, ϕ:: Float64) = μ
∫_ϕ(st:: SphericalZhat, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = μ * (ϕ2 - ϕ1)
(st:: SphericalXX)(μ:: Float64, ϕ:: Float64):: Float64 = (1 - μ ^ 2) * cos(ϕ) ^ 2
∫_ϕ(st:: SphericalXX, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (1 - μ ^ 2) * ((1 / 2) * (ϕ2 - ϕ1) - (1/4) * (sin(2ϕ2) - sin(2ϕ1)))
(st:: SphericalXY)(μ:: Float64, ϕ:: Float64):: Float64 = (1 - μ ^ 2) * cos(ϕ) * sin(ϕ)
∫_ϕ(st:: SphericalXY, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (1 - μ ^ 2) * (1 / 4) * (cos(2ϕ2) - cos(2ϕ1))
(st:: SphericalXZ)(μ:: Float64, ϕ:: Float64):: Float64 = (μ * √(1 - μ ^ 2)) * cos(ϕ)
∫_ϕ(st:: SphericalXZ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (μ * √(1 - μ ^ 2)) * (sin(ϕ1) - sin(ϕ2))
(st:: SphericalYY)(μ:: Float64, ϕ:: Float64):: Float64 = (1 - μ ^ 2) * (sin(ϕ)) ^ 2
∫_ϕ(st:: SphericalYY, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (1 - μ ^ 2) * ((1/2) * (ϕ2 - ϕ1) + (1/4) * (sin(2ϕ2) - sin(2ϕ1)))
(st:: SphericalYZ)(μ:: Float64, ϕ:: Float64):: Float64 = μ * √(1 - μ ^ 2) * sin(ϕ)
∫_ϕ(st:: SphericalYZ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = μ * √(1 - μ ^ 2) * (cos(ϕ2) - cos(ϕ1))
(st:: SphericalZZ)(μ:: Float64, ϕ:: Float64):: Float64 = μ ^ 2
∫_ϕ(st:: SphericalZZ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = μ ^ 2 * (ϕ2 - ϕ1)

const diagonal:: Vector{<: TensorDirection} = SphericalXX() ⊕ SphericalYY() ⊕ SphericalZZ()
const upper_right:: Vector{<: TensorDirection} = reduce(⊕, [SphericalXX(), SphericalXY(), SphericalXZ(), SphericalYY(), SphericalYZ(), SphericalZZ()])


end