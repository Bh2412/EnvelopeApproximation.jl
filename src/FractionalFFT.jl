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

    function FractionalFFTBuffer{M}() where M
        y = zeros(ComplexF64, 2M)
        z = zeros(ComplexF64, 2M)
        return new{M}(y, z)
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
    fft!(buffer.y)
    fft!(buffer.z)
    @. buffer.y *= buffer.z
    ifft!(buffer.y)
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

function fractional_fftfreq(α:: Float64, M:: Int, Δt:: Float64):: AbstractVector{Float64}
    # This assumes the vector v are evenly spaced samples in the range - [0., length(v) * Δt]
    # with endpoints included. 
    γ = α * 2π * Δt # The k resolution
    return γ .* range(0, M - 1)
end

#=
We take as a default the time range to be [0., 1.]
=#
fractional_fftfreq(α:: Float64, M:: Int) = fractional_fftfreq(α, M, 1 / (M - 1))  # The substraction of 1. is due ot the inclusion of end points

end