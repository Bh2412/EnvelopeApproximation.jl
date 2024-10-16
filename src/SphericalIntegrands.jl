abstract type SphericalIntegrand{T} end

function (si:: SphericalIntegrand{T})(μ:: Float64, ϕ:: Float64):: T where T  
    throw(error("Not Implemented"))
end

function (si:: SphericalIntegrand{T})(μϕ:: Tuple{Float64, Float64}):: T where T
    return si(μϕ...)
end

function ∫_ϕ(si:: SphericalIntegrand{T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: T where T
    throw(error("Not Implemented"))
end

function ∫_ϕ(V:: AbstractVector{T}, si:: SphericalIntegrand{AbstractVector{T}}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: AbstractVector{T} where T
    return V .= ∫_ϕ(si, μ, ϕ1, ϕ2)
end

function ∫_ϕ(si:: SphericalIntegrand{T}, μ:: Float64):: T where T
    return ∫_ϕ(si, μ, 0., 2π)
end

function ∫_ϕ(V:: AbstractVector{T}, si:: SphericalIntegrand{AbstractVector{T}}, μ:: Float64):: AbstractVector{T} where T
    return V .= ∫_ϕ(si, μ, 0., 2π)
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

struct SphericalDirectSumIntegrand{K, T} <: SphericalIntegrand{NTuple{K, T}}
    components:: NTuple{K, SphericalIntegrand{T}} 
end

function (ds:: SphericalDirectSumIntegrand{K, T})(μ:: Float64, ϕ:: Float64):: NTuple{K, T} where {K, T}
    ((μ, ϕ), ) .|> ds.components
end

function (ds:: SphericalDirectSumIntegrand{K, T})(V:: Vector{T}, μ:: Float64, ϕ:: Float64):: Vector{T} where {K, T}
    @. V = ((μ, ϕ), ) |> ds.components
    return V
end

function ⊕(si1:: SphericalIntegrand{T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{2, T} where T
    SphericalDirectSumIntegrand{2, T}((si1, si2))
end

function ⊕(si1:: SphericalDirectSumIntegrand{K, T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{K + 1, T} where {K, T}
    SphericalDirectSumIntegrand{K+1, T}(((si1.components..., si2)))
end

function ∫_ϕ(sdsi:: SphericalDirectSumIntegrand{K, T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: NTuple{K, T} where {K, T}
    return ∫_ϕ.(sdsi.components, μ, ϕ1, ϕ2)
end

function ∫_ϕ!(V:: AbstractVector{T}, sdsi:: SphericalDirectSumIntegrand{K, T}, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Vector{T} where {K, T}
    return V .= ∫_ϕ(sdsi, μ, ϕ1, ϕ2)
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

const diagonal:: SphericalDirectSumIntegrand{3, Float64} = SphericalXX() ⊕ SphericalYY() ⊕ SphericalZZ()
const upper_right:: SphericalDirectSumIntegrand{6, Float64} = reduce(⊕, [SphericalXX(), SphericalXY(), SphericalXZ(), SphericalYY(), SphericalYZ(), SphericalZZ()])
