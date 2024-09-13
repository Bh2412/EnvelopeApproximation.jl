module StressEnergyTensorFFT
using EnvelopeApproximation.BubbleBasics
using FastSphericalHarmonics
using StaticArrays
using SpecialFunctions
using SphericalHarmonics
import Base: call, *, ⊕
using StaticArrays

abstract type SphericalIntegrand{T} end

call(si:: SphericalIntegrand{T}, Θ:: Float64, Φ:: Float64):: T = throw(MethodError(call, (si,)))

struct SphericalMultiplicationIntegrand{T} <: SphericalIntegrand{T}
    components:: NTuple{K, SphericalIntegrand} where K
end

call(m:: SphericalMultiplicationIntegrand{T}, Θ:: Float64, Φ:: Float64):: T = prod(c(Θ, Φ) for c in m.components)

*(si1:: SphericalIntegrand{Float64}, si2:: SphericalIntegrand{SVector{K, Float64}}):: SphericalMultiplicationIntegrand{SVector{K, Float64}} where K = SphericalMultiplicationIntegrand{SVector{Float64}}((si1, si2))

struct SphericalDirectSumIntegrand{K, T} <: SphericalIntegrand{SVector{K, T}}
    components:: NTuple{K, Z} where Z <: SphericalIntegral{T}
end

call(ds:: SphericalDirectSumIntegrand{K, T}, Θ:: Float64, Φ:: Float64):: SVector{K, T} = SVector{K, T}((c(Θ, Φ) for c in ds.components)...)

⊕(si1:: SphericalIntegrand{T}, si2:: SphericalIntegrand{T}) = SphericalDirectSumIntegrand{2, T}(SVector{2}(si1, si2))
⊕(si1:: SphericalDirectSumIntegrand{K, T}, si2:: SphericalIntegrand{T}):: SphericalDirectSumIntegrand{K + 1, T} = SphericalDirectSumIntegrand{K+1, T}(((si1.components..., si2)))

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

function Ylm_decomposition!(V:: Array{Float64, 3}, f:: SphericalIntegrand{SVector{K, Float64}}, n:: Int)
    Θ, Φ = sph_points(n)
    Θ = reshape(Θ, :, 1)
    Φ = reshape(Φ, 1, :)
    for (m, ϕ) ∈ enumerate(Φ), (l, θ) ∈ enumerate(Θ)
        V[l, m, :] = f(θ, ϕ)
    end
    for k in 1:K
        @views sph_transform!(V[:, :, k])
    end
end

function Ylm_decomposition(f:: SphericalIntegrand{SVector{K, Float64}}, n:: Int):: Array{Float64, 3}
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
                  k_Φ:: AbstractVector{Float64s}, 
                  n:: Int64, R:: Float64 = 1.)
    KM = Array{ComplexF64, 5}(undef, n, 2 * n - 1, length(k_r), length(k_Θ), length(k_Φ))
    k_matrix!(KM, k_r, k_Θ, k_Φ, n, R)
    return KM
end

function spherical_planewave_decomposition(f:: SphericalIntegrand{Float64}, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 3}
    V = reshape(Ylm_decomposition(f, n), n, :, 1, 1, 1)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return @. $sph_sum(V * KM)
end

function spherical_planewave_decomposition(f:: SphericalIntegrand{SVector{Float64}}, 
                                           n:: Int64, 
                                           k_r:: AbstractVector{Float64}, 
                                           k_Θ:: AbstractVector{Float64}, 
                                           k_Φ:: AbstractVector{Float64}):: Array{ComplexF64, 3}
    V = reshape(Ylm_decomposition(f, n), n, :, 1, 1, 1)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return @. $sph_sum(V * KM)
end

function td_integrand(θ:: Float64, ϕ:: Float64, s:: Symbol):: Float64
    if s ≡ :trace
        return 1.
    elseif s ≡ :x
        return cos(ϕ) * sin(θ)
    elseif s ≡ :y
        return sin(ϕ) * sin(θ)
    elseif s ≡ :z
        return cos(θ)
    end
end

function td_integrand(θ:: Float64, ϕ:: Float64, td:: Tuple{Symbol, Symbol}):: Float64 
    td ≡ (:x, :x) && return cos(ϕ) ^ 2 * (sin(θ) ^ 2)
    td ≡ (:y, :y) && return sin(ϕ) ^ 2 * (sin(θ) ^ 2)
    td ≡ (:z, :z) && return cos(θ) ^ 2
    ((td ≡ (:x, :y)) | (td ≡ (:y, :x))) && return cos(ϕ) * sin(ϕ) * (sin(θ) ^ 2)
    return td_integrand(θ, ϕ, td[1]) * td_integrand(θ, ϕ, td[2])
end

end