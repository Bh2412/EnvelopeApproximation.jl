module StressEnergyTensorFFT
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.SurfaceTesselation
using FastSphericalHarmonics
using SphericalHarmonicModes
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
    SVector{K, T}(invoke.(ds.components, Tuple{Float64, Float64}, Θ, Φ))
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
        V[:, :, k] .= sph_transform!(V[:, :, k])
    end
end

function Ylm_decomposition(f:: SphericalIntegrand{SVector{K, Float64}}, n:: Int):: Array{Float64, 3} where K
    V = Array{Float64, 3}(undef, n, 2 * n - 1, K)
    Ylm_decomposition!(V, f, n)
    return V
end

function sph_sum(v:: Matrix{T}):: T where T
    @assert size(v)[2] = 2 * size(v)[1] - 1
    return sum(v[sph_mode(l, m)] for (l, m) ∈ LM(0:lmax))
end

function sph_sum(V:: Array{T, 5}):: Array{T, 3} where T
    lmax = sph_lmax(size(V)[1])
    @views sum(V[sph_mode(l, m), :, :, :] for (l, m) ∈ LM(0:lmax))
end

function sph_sum(V:: Array{T, 6}):: Array{T, 4} where T
    lmax = sph_lmax(size(V)[1])
    @views sum(V[sph_mode(l, m), :, :, :, :] for (l, m) ∈ LM(0:lmax))
end

function sph_dot(V:: Array{Float64, 2}, KM:: Array{ComplexF64, 5}, n:: Int):: Array{ComplexF64, 3}
    res = zeros(size(KM[3:end]))
    @inbounds for (l, m) ∈ LM(0:sph_lmax(n))
        @views @. res +=  V[sph_mode(l, m)] * KM[sph_mode(l, m), :, :, :]
    end
    return res
end

function sph_dot(V:: Array{Float64, 3}, KM:: Array{ComplexF64, 5}, n:: Int):: Array{ComplexF64, 4}
    SV = size(V)
    SKM = size(KM)
    @assert SV[1:2] == SKM[1:2] == (n, 2 * n - 1)
    res = zeros(ComplexF64, SKM[3:end]..., SV[end])
    V = reshape(V, n, 2 * n - 1, 1, 1, 1, size(V)[end])
    KM = reshape(KM, size(KM)..., 1)
    @inbounds for (l, m) ∈ LM(0:sph_lmax(n))
        @views @. res +=  V[$sph_mode(l, m), :, :, :, :] * KM[$sph_mode(l, m), :, :, :, :]
    end
    return res
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
        @inbounds for n ∈ eachindex(k_r), (l, m) ∈ LM(0:lmax), 
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
Computes an integral of the form e^(-i * k⋅ r) over a sphere's surface, for various values of k, using a spherical FFT algorithm.
"""
function spherical_planewave_decomposition(f:: SphericalIntegrand{Float64}, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 3}
    V = Ylm_decomposition(f, n)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return sph_dot(V, KM, n)
end

function spherical_planewave_decomposition(f:: SphericalIntegrand{SVector{K, Float64}}, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 4} where K
    V = Ylm_decomposition(f, n)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return sph_dot(V, KM, n)
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
TensorDirection = Union{SphericalTrace, SphericalXhat, 
                        SphericalYhat, SphericalZhat, 
                        SphericalXX, SphericalXY, SphericalXZ, 
                        SphericalYY, SphericalZZ}

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

struct BubbleIntersectionSurfaceIntegrand{K} <: SphericalIntegrand{SVector{K, Float64}}
    bi:: BubbleIntersection
    R:: Float64
    ΔV:: Float64
    tds:: NTuple{K, TensorDirection}
end

function (bii:: BubbleIntersectionSurfaceIntegrand{K})(Θ:: Float64, Φ:: Float64):: SVector{K, Float64} where K
    ∉(cos(Θ), Φ, bii.bi) && return SVector{K, Float64}(zeros(K))
    return @. (bii.ΔV / 3. * bii.R^3) * $SVector(invoke(bii.tds, Tuple{Float64, Float64}, Θ, Φ))
end 

struct BubbleIntersectionPotentialIntegrand <: SphericalIntegrand{SVector{3, Float64}}
    bi:: BubbleIntersection
    R:: Float64
    ΔV:: Float64
    tds:: NTuple{3, TensorDirection}
    function BubbleIntersectionPotentialIntegrand(bi:: BubbleIntersection, R:: Float64, ΔV:: Float64)
        return new(bi, R, ΔV, (SphericalXhat(), SphericalYhat(), SphericalZhat()))
    end
end

function (bip:: BubbleIntersectionPotentialIntegrand)(Θ:: Float64, Φ:: Float64):: SVector{3, Float64}
    ∉(cos(Θ), Φ, bii.bi) && return SVector{3, Float64}(zeros(3))
    return @. (-bip.ΔV * R^2) * $SVector(invoke(bii.tds, Tuple{Float64, Float64}, Θ, Φ))
end

function dot(p:: Vec3, k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64):: Float64
    return  k_r * (p[1] * sin(k_Θ) * cos(k_Φ) + p[2] * sin(k_Θ) * sin(k_Φ) + p[3] * cos(k_Θ))
end

function translation_phase(p:: Point3, 
                           k_r:: AbstractVector{Float64}, 
                           k_Θ:: AbstractVector{Float64}, 
                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 3}
    k_r = reshape(k_r, :, 1, 1)
    k_Θ = reshape(k_Θ, 1, :, 1)
    k_Φ = reshape(k_Φ, 1, 1, :)
    return @. exp(-im * dot((p.coordinates, ), k_r, k_Θ, k_Φ))
end

function surface_integral(k_r:: AbstractVector{Float64}, 
                          k_Θ:: AbstractVector{Float64}, 
                          k_Φ:: AbstractVector{Float64}, 
                          n:: Int64,
                          bubbles:: Bubbles, 
                          tensor_directions:: NTuple{K, TensorDirection} where K,
                          bubble_intersections:: Dict{Int64, BubbleIntersection}, 
                          ΔV:: Float64 = 1.):: Array{ComplexF64, 4}
    V = zeros(ComplexF64, length(k_r), length(k_Θ), length(k_Φ), length(tensor_directions))
    for (bubble_index, bi) in bubble_intersections
        bubble_integrand = BubbleIntersectionSurfaceIntegrand{length(tensor_directions)}(bi, bubbles[bubble_index].radius, 
                                                                                         ΔV, tensor_directions)
        @. V += $spherical_planewave_decomposition(bubble_integrand, n, k_r, k_Θ, k_Φ) * 
                $reshape($translation_phase(bubbles[bubble_index].center, k_r, k_Θ, k_Φ), $length(k_r), $length(k_Θ), $length(k_Φ), 1)
    end
    return V
end

function surface_integral(k_r:: AbstractVector{Float64}, 
                          k_Θ:: AbstractVector{Float64}, 
                          k_Φ:: AbstractVector{Float64}, 
                          n:: Int64,
                          bubbles:: Bubbles, 
                          tensor_directions:: NTuple{K, TensorDirection} where K,
                          ϕ_resolution:: Float64, μ_resolution:: Float64, 
                          ΔV:: Float64 = 1.):: Array{ComplexF64, 4}
    bis = bubble_intersections(ϕ_resolution, μ_resolution, bubbles)
    return surface_integral(k_r, k_Θ, k_Φ, n, bubbles, tensor_directions, bis, ΔV)
end

function surface_integral(k_r:: AbstractVector{Float64}, 
    k_Θ:: AbstractVector{Float64}, 
    k_Φ:: AbstractVector{Float64}, 
    n:: Int64,
    bubbles:: Bubbles, 
    tensor_directions:: NTuple{K, TensorDirection} where K,
    n_ϕ:: Int64, n_μ:: Int64, 
    ΔV:: Float64 = 1.):: Array{ComplexF64, 4}
    return surface_integral(k_r, k_Θ, k_Φ, n, bubbles, tensor_directions, 2π / n_ϕ, 2. / n_μ, ΔV)
end

end