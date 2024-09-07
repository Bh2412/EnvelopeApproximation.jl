module StressEnergyTensorFFT
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubbleBasics: ⋅
using EnvelopeApproximation.BubblesEvolution
using StaticArrays
using FastSphericalHarmonics
import SpecialFunctions: besselj
using LinearAlgebra

function td_integrand(x:: SVector{2, Float64}, s:: Symbol):: Float64
    if s ≡ :trace
        return 1.
    elseif s ≡ :x
        return cos(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :y
        return sin(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :z
        return x[2]
    end
end

function td_integrand(x:: SVector{2, Float64}, td:: Tuple{Symbol, Symbol}):: Float64 
    td ≡ (:x, :x) && return cos(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:y, :y) && return sin(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:z, :z) && return x[2] ^ 2
    ((td ≡ (:x, :y)) | (td ≡ (:y, :x))) && return cos(x[1]) * sin(x[1]) * (1 - x[2] ^ 2)
    return td_integrand(x, td[1]) * td_integrand(x, td[2])
end

function td_integrand(ϕ:: Float64, μ:: Float64, td:: Symbol)
    return td_integrand(SVector{2, Float64}([ϕ, μ]), td)
end

function td_integrand(ϕ:: Float64, μ:: Float64, td:: Tuple{Symbol, Symbol})
    return td_integrand(SVector{2, Float64}([ϕ, μ]), td)
end

sphericalbesselj(ν:: Float64, x:: Float64) = √(π/2x)*besselj(ν+1/2, x)
sphericalbesselj(l:: Int64, x:: Float64) = √(π/2x)*besselj(l * (l + 1) + 1/2, x)

trace(θ:: Float64, φ:: Float64) = 1.

n = 10

θs, φs = sph_points(n)

V = Matrix{Float64}(undef, n, 2n -1)

V .= 1.

R = 4.

sph_transform!(V)

function k_matrix(ks:: Vector{Vec3}, n:: Int64):: Array{ComplexF64, 3}
    KM = Array{ComplexF64, 3}(undef, length(ks), n, 2n - 1)
    lm = sph_lmax(n)
    @inbounds for (i, k) ∈ enumerate(ks)  
        @inbounds for l ∈ 0:lm, m ∈ -l:l 
            KM[i, sph_mode(l, m)] = 4π * (-i) ^ l * sphericalbesselj(l, k[1] * R)
        end
        sph_evaluate!(KM[i, :, :])
    end
    return KM
end

function plane_wave_decompoition(f:: Function, n:: Int, ks:: Vector{Vec3}):: Vector{ComplexF64}
    θs, φs = sph_points(n)
    lm  = sph_lmax(n)
    V = f.(reshape(θs, :, 1), reshape(φs, 1, :))
    sph_transform!(V)
    KM = k_matrix(ks, n)   
    Res = zeros(ComplexF64, length(ks))
    @inbounds for i ∈ eachindex(ks)  
        @inbounds for l ∈ 0:lm, m ∈ -l:l 
            Res[i] += KM[i, sph_mode(l, m)] * V[sph_mode(l, m)]
        end
    end
    return Res
end

end