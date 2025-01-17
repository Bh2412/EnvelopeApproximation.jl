using EnvelopeApproximation.BubbleBasics
using FastSphericalHarmonics
using StaticArrays
using SpecialFunctions
using SphericalHarmonics

# For a single bubble

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

function Ylm_decomposition!(V:: Matrix{Float64}, f:: Function, n:: Int)
    Θ, Φ = sph_points(n)
    Θ = reshape(Θ, :, 1)
    Φ = reshape(Φ, 1, :)
    @. V = f(Θ, Φ)
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
        return -1.
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
        KM[sph_mode(l, m)] *= cl[l + 1] * Ylm[(l, m)]
    end
end

function k_matrix!(KM:: Union{Matrix{ComplexF64}, Matrix{Float64}}, k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64, n:: Int64; R:: Float64 = 1.)
    @assert size(KM) == (n, 2 * n - 1)
    l = 0:1:sph_lmax(n) 
    cl = @. sphericalbesselj(l, k_r * R) * phase(l) * (4π)
    Ylm = computeYlm(k_Θ, k_Φ; lmax=last(l), SHType = SphericalHarmonics.RealHarmonics())
    @inbounds for l ∈ l, m ∈ -l:l
        KM[sph_mode(l, m)] = cl[l + 1] * Ylm[(l, m)]
    end
end

function computeYlm!(KM:: Union{Matrix{ComplexF64}, Matrix{Float64}}, k_Θ:: Float64, k_Φ:: Float64, n:: Int64)
    Ylm = computeYlm(k_Θ, k_Φ; lmax=n-1, SHType = SphericalHarmonics.RealHarmonics())
    @inbounds for l ∈ 0:l, m ∈ -l:l
        KM[sph_mode(l, m)] = cl[l + 1] * Ylm[(l, m)]
    end
end

function k_matrix!(KM:: Array{ComplexF64, 5}, k_r:: StepRangeLen, k_Θ:: StepRangeLen, k_Φ:: StepRangeLen, n:: Int64; R:: Float64 = 1.)
    lmax = sph_lmax(n)
    φl = @. phase(0:lmax) * 4π
    c_l_k_r = @. sphericalbesselj($reshape((0:lmax), :, 1), $reshape(k_r * R, 1, :))
    @inbounds for (j, k_ϕ) ∈ enumerate(k_Φ), (i, k_θ) ∈ enumerate(k_Θ)
        Ylm = computeYlm(k_θ, k_ϕ; lmax=lmax, SHType = SphericalHarmonics.RealHarmonics())
        @inbounds for n ∈ eachindex(k_r), l ∈ 0:lmax, m ∈ -l:l, 
            KM[sph_mode(l, m), n, i, j] = (Ylm[(l, m)] * c_l_k_r[l + 1, n]) * φl[l + 1]
        end
    end
end

function k_matrix(k_r:: StepRangeLen, k_Θ:: StepRangeLen, k_Φ:: StepRangeLen, n:: Int64; R:: Float64 = 1.)
    KM = Array{ComplexF64, 5}(undef, n, 2 * n - 1, length(k_r), length(k_Θ), length(k_Φ))
    k_matrix!(KM, k_r, k_Θ, k_Φ, n; R=R)
    return KM
end

function k_matrix(k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64, n:: Int64; R:: Float64 = 1.)
    KM = zeros(ComplexF64, n, 2 * n - 1)
    k_r
    k_matrix!(KM, k_r, k_Θ, k_Φ, n; R=R)
    return KM
end


function sph_sum(v:: Matrix{T}):: T where T
    @assert size(v)[2] = 2 * size(v)[1] - 1
    return sum(v[sph_mode(l, m)] for l ∈ 0:1:lmax for m ∈ -l:1:l)
end

function sph_sum(V:: Array{T, 5}):: Array{T, 3} where T
    lmax = sph_lmax(size(V)[1])
    @views sum(V[sph_mode(l, m), :, :, :] for l ∈ 0:lmax for m ∈ -l:l)
end

function sph_dot(v1:: Matrix{T}, v2:: Matrix{T}):: T where T
    @assert size(v1) == size(v2)
    @assert size(v1)[2] = 2 * size(v1)[1] - 1
    lmax = sph_lmax(size(v1))
    return sum((v1[sph_mode(l, m)] * v2[sph_mode(l, m)]) for l ∈ 0:1:lmax for m ∈ -l:1:l)
end

function spherical_planewave_decomposition(f:: Function, n:: Int64, k_r:: Float64, k_Θ:: Float64, k_Φ:: Float64):: ComplexF64
    V = Ylm_decomposition(f, n)
    k_matrix_mul!(V, k_r, k_Θ, k_Φ, n)
    return sum(V)
end

function spherical_planewave_decomposition(f:: Function, n:: Int64, k_r:: StepRangeLen, k_Θ:: StepRangeLen, k_Φ:: StepRangeLen):: Array{ComplexF64, 3}
    V = reshape(Ylm_decomposition(f, n), n, :, 1, 1, 1)
    KM = k_matrix(k_r, k_Θ, k_Φ, n)
    return @. $sph_sum(V * KM)
end


using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.StressEnergyTensor
using Plots

R = 1.
bubbles = Bubbles([Bubble(Point3(0., 0., 0.), R)])

k_0 = 2π / R
ks = (k_0 / 10):(k_0 / 10):(k_0 * 10)
k_vecs = (x -> Vec3(0., 0., x)).(ks)

function integrand(θ:: Float64, ϕ:: Float64, tensor_direction:: TensorDirection)
    return td_integrand(θ, ϕ, tensor_direction) * (ΔV / 3. * R ^ 3)
end

ΔV = 1.
analytic_T_ii = @. ((ΔV * 4π / 3) * (R ^ 3)) * sin(ks * R) / (ks * R)  
plot(ks, analytic_T_ii)
sph_T_ii = spherical_planewave_decomposition((θ, ϕ) -> integrand(θ, ϕ, :trace), 
                                             10, ks, 0.:0., 0.:0.) .|> real
@time spherical_planewave_decomposition((θ, ϕ) -> integrand(θ, ϕ, :trace), 
50, ks, 0.:0., 0.:0.) .|> real
@time numerical_T_ii = surface_integral(k_vecs, bubbles, Vector{TensorDirection}([:trace]), 50, 50, ΔV)

analytic_T_xx = @. 4π / 3 * ΔV  / (ks ^ 3) * (sin(ks * R) - (ks * R) * cos(ks * R))
@time sph_T_xx = spherical_planewave_decomposition((θ, ϕ) -> integrand(θ, ϕ, (:x, :x)), 
10, ks, 0.:0., 0.:0.) .|> real |> x -> reshape(x, :)
@time numeric_T_xx = surface_integral(k_vecs, bubbles, Vector{TensorDirection}([(:x, :x)]), 50, 50, ΔV)

analytic_T_zz = @.(4π / 3 * ΔV * (1 / (ks ^ 3)) * (2 * ks * R * cos(ks * R) + (R^2 * ks^2 - 2) * sin(ks * R)))
@time sph_T_zz = spherical_planewave_decomposition((θ, ϕ) -> integrand(θ, ϕ, (:z, :z)), 100, 
ks, 0.:0., 0.:0.) .|> real |> x -> reshape(x, :)
