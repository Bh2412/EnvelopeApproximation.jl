module FractionalFFT
using FFTW

#=
Following "A Fast method for the numerical evaluation of Continuous Fourier and Laplace Transforms" - David H. Bailey and Paul N. Swartzbauer
Specifically, equation 18.
Note that indices run from 1 here and 0 there.
=#

export FractionalFFTBuffer

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

export fractional_fft

function fractional_fft(v:: Vector{Float64}, α:: Float64):: AbstractVector{ComplexF64}
    buffer = FractionalFFTBuffer{length(v)}()
    return fractional_fft(v, α, buffer)
end
    
export fractional_fftfreq

function fractional_fftfreq(γ:: Float64, M:: Int)
    # γ is the k resolution, usually relate to the α parameter in the fractional FFT (see above mentioned reference)
    return γ .* range(0, M-1)
end

function fractional_fftfreq(α:: Float64, M:: Int, Δt:: Float64):: AbstractVector{Float64}
    # This assumes the vector v are evenly spaced samples in the range - [0., length(v) * Δt]
    # with endpoints included. 
    γ = α * 2π * Δt # The k resolution
    return fractional_fftfreq(γ, M)
end

end