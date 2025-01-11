using FFTW
using BenchmarkTools

#=
Following "A Fast method for the numerical evaluation of Continuous Fourier and Laplace Transforms" - David H. Bailey and Paul N. Swartzbauer
Specifically, equation 18.
Note that indices run from 1 here and 0 there.
=#

struct FractionalFFTBuffer{M}
    y:: Vector{ComplexF64}
    z:: Vector{ComplexF64}
    P:: FFTW.cFFTWPlan{ComplexF64, -1, true, 1, Tuple{Int64}}
    inverse_P:: AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, Tuple{Int64}}, Float64}

    function FractionalFFTBuffer{M}() where M
        y = zeros(ComplexF64, 2M)
        z = zeros(ComplexF64, 2M)
        P = plan_fft!(y)
        inverse_P = inv(P)
        return new{M}(y, z, P, inverse_P)
    end
end

function prepare!(v:: Vector{Float64}, α:: Float64, buffer:: FractionalFFTBuffer{M}) where M
    @assert M == length(v) "FractionalFFTBuffer size must be 2*N."
    @inbounds for j in 0:(M-1)
        j̃ = j + 1
        buffer.y[j̃] = v[j̃] * cispi(-α * (j^2))
        buffer.z[j̃] = cispi(α * (j^2))
    end
    @inbounds for j in M:(2M - 1)
        j̃ = j + 1
        buffer.y[j̃] = 0.
        buffer.z[j̃] = cispi(α * ((j - 2M) ^ 2))
    end
end

function fractional_fft(v:: Vector{Float64}, α:: Float64, buffer:: FractionalFFTBuffer{M}):: AbstractVector{ComplexF64} where M
    @assert M == length(v) "FractionalFFTBuffer size must be 2*M."
    prepare!(v, α, buffer)
    buffer.P * buffer.y
    buffer.P * buffer.z
    @. buffer.y *= buffer.z
    buffer.inverse_P * buffer.y
    @inbounds for k in 0:(M-1)
        k̃ = k + 1
        buffer.y[k̃] *= cispi(-α * (k ^ 2))
    end
    @views return buffer.y[1:M]
end

function fractional_fft(v:: Vector{Float64}, α:: Float64):: AbstractVector{ComplexF64}
    buffer = FractionalFFTBuffer{length(v)}()
    return fractional_fft(v, α, buffer)
end

function direct_fractional_fft(v:: Vector{Float64}, α:: Float64):: Vector{ComplexF64}
    M = length(v)
    V = Vector{ComplexF64}(undef, M)
    for k in 0:(M-1)
        k̃ = k + 1
        V[k̃] = sum(_v * cispi(-2 * (j - 1) * k * α) for (j, _v) in enumerate(v), init=zero(ComplexF64))
    end
    return V
end

N = 2 ^ 7
a, b = 0., 1.
v = sin.(range(a, b, N))
α = 1 / N
# @btime $direct_fractional_fft($v, $α)
buffer = FractionalFFTBuffer{N}()
# @btime $fractional_fft($v, $α, $buffer)
# @profview for _ in 1:10_000 fractional_fft(v, α, buffer) end
# @btime fft($v)
fractional_fft(v, α) ≈ fft(v) 
fractional_fft(v, α) ≈ direct_fractional_fft(v, α)
α = 0.0001
fractional_fft(v, α) ≈ direct_fractional_fft(v, α)

