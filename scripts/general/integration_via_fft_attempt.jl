using EnvelopeApproximation.BubbleBasics
using FastSphericalHarmonics
using StaticArrays
using SpecialFunctions
using SphericalHarmonics

# For a single bubble

function td_integrand(ϕ:: Float64, μ:: Float64, s:: Symbol):: Float64
    if s ≡ :trace
        return 1.
    elseif s ≡ :x
        return cos(ϕ) * √(1 - μ ^ 2)
    elseif s ≡ :y
        return sin(ϕ) * √(1 - μ ^ 2)
    elseif s ≡ :z
        return μ
    end
end

function td_integrand(ϕ:: Float64, μ:: Float64, td:: Tuple{Symbol, Symbol}):: Float64 
    td ≡ (:x, :x) && return cos(ϕ) ^ 2 * (1 - μ ^ 2)
    td ≡ (:y, :y) && return sin(ϕ) ^ 2 * (1 - μ ^ 2)
    td ≡ (:z, :z) && return μ ^ 2
    ((td ≡ (:x, :y)) | (td ≡ (:y, :x))) && return cos(ϕ) * sin(ϕ) * (1 - μ ^ 2)
    return td_integrand(ϕ, μ, td[1]) * td_integrand(ϕ, μ, td[2])
end

function Ylm_decomposition!(V:: Matrix{Float64}, f:: Function, n:: Int)
    Θ, Φ = sph_points(n)
    Θ = reshape(Θ, :, 1)
    Φ = reshape(Φ, 1, :)
    V .= f.(Θ, Φ)
    sph_transform!(V)
end

function Ylm_decomposition(f:: Function, n:: Int):: Matrix{Float64}
    V = Matrix{Float64}(undef, n, 2 * n  - 1)
    return Ylm_decomposition!(V, f, n)
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function phase(l:: Int64):: ComplexF64
    l = l % 4
    if l == 0
        return 1.
    elseif l == 1
        return im
    elseif l == 2
        return -1
    elseif l == 3
        return -im
    end
end

function k_matrix_mul!(KM:: Union{Matrix{ComplexF64}, Matrix{Float64}}, k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64, n:: Int64; R:: Float64 = 1.)
    @assert size(KM) == (n, 2 * n - 1)
    l = 0:1:sph_lmax(n) 
    cl = @. sphericalbesselj(l, k_r * R) * phase(l) * (4π)
    Ylm = computeYlm(k_Θ, k_Φ; lmax=last(l), SHType = SphericalHarmonics.RealHarmonics())
    @inbounds for l ∈ l, m ∈ -l:1:l
        KM[l + 1, m + l + 1] *= cl[l + 1] * Ylm[(l, m)]
    end
end

function k_matrix(k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64, n:: Int64; R:: Float64 = 1.)
    KM = one(ComplexF64, n, 2 * n - 1)
    k_matrix_mul!(KM, k_r, k_Θ, k_Φ, n; R=R)
    return KM
end

function spherical_planewave_decomposition(f:: Function, n:: Int64, k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64):: ComplexF64
    V = Ylm_decomposition(f, n)
    k_matrix_mul!(V, k_r, k_Θ, k_Φ, n)
    return sum(V)
end
