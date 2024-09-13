module StressEnergyTensorFFT
using EnvelopeApproximation.BubbleBasics
using FastSphericalHarmonics
using StaticArrays
using SpecialFunctions
using SphericalHarmonics
import Base: *
using StaticArrays

abstract type SphericalIntegrand{T} end

function (si:: SphericalIntegrand{T})(Θ:: Float64, Φ:: Float64):: T where T  
    throw(error("Not Implemented"))
end

struct SphericalMultiplicationIntegrand{T} <: SphericalIntegrand{T}
    components:: NTuple{K, SphericalIntegrand} where K
end

function (m:: SphericalMultiplicationIntegrand{T})(Θ:: Float64, Φ:: Float64):: T where T
    prod(c(Θ, Φ) for c in m.components)
end

function *(si1:: SphericalIntegrand{Float64}, si2:: SphericalIntegrand{SVector{K, Float64}}):: SphericalMultiplicationIntegrand{SVector{K, Float64}} where K 
    SphericalMultiplicationIntegrand{SVector{Float64}}((si1, si2))
end

struct SphericalDirectSumIntegrand{K, T} <: SphericalIntegrand{SVector{K, T}}
    components:: NTuple{K, Z} where Z <: SphericalIntegrand{T}
end

function (ds:: SphericalDirectSumIntegrand{K, T})(Θ:: Float64, Φ:: Float64):: SVector{K, T} where {K, T}
    SVector{K, T}((c(Θ, Φ) for c in ds.components)...)
end

function ⊕(si1:: SphericalIntegrand{T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{2, T} where T
    SphericalDirectSumIntegrand{2, T}((si1, si2))
end

function ⊕(si1:: SphericalDirectSumIntegrand{K, T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{K + 1, T} where {K, T}
    SphericalDirectSumIntegrand{K+1, T}(((si1.components..., si2)))
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function Ylm_decomposition!(V:: Matrix{Float64}, f:: SphericalIntegrand{Float64}, n:: Int)
    Θ, Φ = sph_points(n)
    Θ = reshape(Θ, :, 1)
    Φ = reshape(Φ, 1, :)
    @. V = f(Θ, Φ)
    sph_transform!(V)
end

function Ylm_decomposition(f:: SphericalIntegrand{Float64}, n:: Int):: Matrix{Float64}
    V = Matrix{Float64}(undef, n, 2 * n  - 1)
    return Ylm_decomposition!(V, f, n)
end

function Ylm_decomposition!(V:: Array{Float64, 3}, f:: SphericalIntegrand{SVector{K, Float64}}, n:: Int) where K
    Θ, Φ = sph_points(n)
    Θ = reshape(Θ, :, 1)
    Φ = reshape(Φ, 1, :)
    @inbounds for (m, ϕ) ∈ enumerate(Φ), (l, θ) ∈ enumerate(Θ)
        V[l, m, :] = f(θ, ϕ)
    end
    @inbounds for k in 1:K
        @views sph_transform!(V[:, :, k])
    end
end

function Ylm_decomposition(f:: SphericalIntegrand{SVector{K, Float64}}, n:: Int):: Array{Float64, 3} where K
    V = Array{3, Float64}(undef, n, 2 * n - 1, K)
    Ylm_decomposition!(V, f, n)
    return V
end

function sph_sum(v:: Matrix{T}):: T where T
    @assert size(v)[2] = 2 * size(v)[1] - 1
    return sum(v[sph_mode(l, m)] for l ∈ 0:1:lmax for m ∈ -l:1:l)
end

function sph_sum(V:: Array{T, 5}):: Array{T, 3} where T
    lmax = sph_lmax(size(V)[1])
    @views sum(V[sph_mode(l, m), :, :, :] for l ∈ 0:lmax for m ∈ -l:l)
end

function k_matrix!(KM:: Array{ComplexF64, 5}, 
                   k_r:: AbstractVector{Float64}, 
                   k_Θ:: AbstractVector{Float64}, 
                   k_Φ:: AbstractVector{Float64}, 
                   n:: Int64, 
                   R:: Float64 = 1.)
    lmax = sph_lmax(n)
    φl = @. (im ^ (0:lmax)) * 4π
    c_l_k_r = @. sphericalbesselj($reshape((0:lmax), :, 1), $reshape(k_r * R, 1, :))
    @inbounds for (j, k_ϕ) ∈ enumerate(k_Φ), (i, k_θ) ∈ enumerate(k_Θ)
        Ylm = computeYlm(k_θ, k_ϕ; lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
        @inbounds for n ∈ eachindex(k_r), l ∈ 0:lmax, m ∈ -l:l, 
            KM[sph_mode(l, m), n, i, j] = (Ylm[(l, m)] * c_l_k_r[l + 1, n]) * φl[l + 1]
        end
    end
end

function k_matrix(k_r:: AbstractVector{Float64}, 
                  k_Θ:: AbstractVector{Float64}, 
                  k_Φ:: AbstractVector{Float64}, 
                  n:: Int64, R:: Float64 = 1.)
    KM = Array{ComplexF64, 5}(undef, n, 2 * n - 1, length(k_r), length(k_Θ), length(k_Φ))
    k_matrix!(KM, k_r, k_Θ, k_Φ, n, R)
    return KM
end

export spherical_planewave_decomposition
"""
Computes an integral of the form $e^(-i * k⋅ r) f$ over a sphere's surface, for various values of k, using a spherical FFT algorithm.
"""
function spherical_planewave_decomposition(f:: SphericalIntegrand{Float64}, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 3}
    V = reshape(Ylm_decomposition(f, n), n, :, 1, 1, 1)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return @. $sph_sum(V * KM)
end

function spherical_planewave_decomposition(f:: SphericalIntegrand{SVector{K, Float64}} where K, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 4}
    V = reshape(Ylm_decomposition(f, n), n, 2 * n - 1, 1, 1, 1, K)
    KM = reshape(k_matrix(k_r, k_Θ, k_Φ, n), n, 2 * n - 1, length(k_r), length(k_Θ), length(k_Φ), 1)
    return @. $sph_sum(V * KM)
end

struct SphericalTrace <: SphericalIntegrand{Float64} end
struct SphericalXhat <: SphericalIntegrand{Float64} end
struct SphericalYhat <: SphericalIntegrand{Float64} end
struct SphericalZhat <: SphericalIntegrand{Float64} end
struct SphericalXX <: SphericalIntegrand{Float64} end
struct SphericalXY <: SphericalIntegrand{Float64} end
SphericalYX = SphericalXY
struct SphericalXZ <: SphericalIntegrand{Float64} end
SphericalZX = SphericalXZ
struct SphericalYY <: SphericalIntegrand{Float64} end
struct SphericalYZ <: SphericalIntegrand{Float64} end
struct SphericalZZ <: SphericalIntegrand{Float64} end


(st:: SphericalTrace)(Θ:: Float64, Φ:: Float64):: Float64 = 1.
(st:: SphericalXhat)(Θ:: Float64, Φ:: Float64):: Float64 = sin(Θ) * cos(Φ)
(st:: SphericalYhat)(Θ:: Float64, Φ:: Float64):: Float64 = sin(Θ) * sin(Φ)
(st:: SphericalZhat)(θ:: Float64, Φ:: Float64) = cos(θ)
(st:: SphericalXX)(Θ:: Float64, Φ:: Float64):: Float64 = (sin(Θ) * cos(Φ)) ^ 2
(st:: SphericalXY)(Θ:: Float64, Φ:: Float64):: Float64 = (sin(Θ) ^ 2) * cos(Φ) * sin(Φ)
(st:: SphericalXZ)(Θ:: Float64, Φ:: Float64):: Float64 = cos(Θ) * sin(Θ) * cos(Φ)
(st:: SphericalYY)(Θ:: Float64, Φ:: Float64):: Float64 = (sin(Θ) * sin(Φ)) ^ 2
(st:: SphericalYZ)(Θ:: Float64, Φ:: Float64):: Float64 = cos(Θ) * sin(Θ) * sin(Φ)
(st:: SphericalZZ)(Θ:: Float64, Φ:: Float64):: Float64 = cos(Θ) ^ 2



end