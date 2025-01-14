module ChebyshevCFT

using Bessels
import Bessels: besselj!
using LinearAlgebra
using FastTransforms

# Following chapter 2.10.5 in "methods of numerical integration"

struct ChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs0_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}

    function ChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs0_buffer = Vector{Float64}(undef, N)
        bessels_buffer = Vector{Float64}(undef, N)
        return new{N}(points, coeffs0_buffer, bessels_buffer)
    end
end

translation(a:: Float64, b:: Float64):: Float64 = (a + b) / 2
scale(a:: Float64, b:: Float64):: Float64 = (b - a) / 2
u(x:: Float64, scale:: Float64, translation:: Float64):: Float64 = (x - translation) / scale
inverse_u(u:: Float64, scale:: Float64, translation:: Float64):: Float64 = translation + scale * u
inverse_chebyshev_weight(u:: Float64):: Float64 = sqrt(1 - u ^ 2)

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: ChebyshevPlan{N}):: AbstractVector{Float64} where N
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in enumerate(chebyshev_plan.points)
        chebyshev_plan.coeffs0_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    return chebyshev_plan.coeffs0_buffer
end

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, chebyshev_plan:: ChebyshevPlan{N}):: Vector{Float64} where N
    return chebyshevtransform!(values!(f, a, b, chebyshev_plan), Val(1))
end

# Following equation 2.10.5.1 and 2.10.5.2 in "methods of numerical integration"

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: ChebyshevPlan{N}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: ComplexF64 where N
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N-1), k̃)
    c, s = 0., 0.
    @inbounds for ((ĩ, coeff), bessel_value) in zip(enumerate(chebyshev_plan.coeffs0_buffer), chebyshev_plan.bessels_buffer)
        i = ĩ - 1
        l = i ÷ 2
        if iseven(i)
            c += ((-1) ^ l) * π * coeff * bessel_value
        else
            s += ((-1) ^ l) * π * coeff * bessel_value
        end
    end
    return (c - im * s) * cis(-k * translation) * scale
end

function fourier_modes!(ks:: AbstractVector{Float64}, 
                        f, a:: Float64, b:: Float64, 
                        chebyshev_plan:: ChebyshevPlan{N}, 
                        buffer:: Vector{ComplexF64}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    @inbounds for (j, k) in enumerate(ks)
        buffer[j] = fourier_mode(k, chebyshev_plan, a, t)
    end
    return buffer
end

function add_fourier_modes!(ks:: AbstractVector{Float64}, 
                            f, a:: Float64, b:: Float64, 
                            chebyshev_plan:: ChebyshevPlan{N}, 
                            buffer:: Vector{ComplexF64}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    @inbounds for (j, k) in enumerate(ks)
        buffer[j] += fourier_mode(k, chebyshev_plan, a, t)
    end
end

function fourier_modes(ks:: AbstractVector{Float64}, 
                       f, a:: Float64, b:: Float64, chebyshev_plan:: ChebyshevPlan{N}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    return fourier_mode.(ks, (chebyshev_plan, ), (a, ), (t, ))
end

function mul_x(N:: Int):: Matrix{Float64}
    # This is based on the Chebyshev multiplication identity found in wikipedia
    M = zeros(Float64, (N+1, N))
    M[1, 2] = 1. / 2
    @views M[2, [1, 3]] .= (1., 1. / 2)
    for j̃ in 3:(N-1)
        @views M[j̃, [j̃ - 1, j̃ + 1]] .= 1. / 2
    end
    M[N, N-1] = 1. / 2
    M[N + 1, N] = 1. / 2
    return M
end

function mul_x_squared(N:: Int):: Matrix{Float64}
    # This is based on the Chebyshev multiplication identity found in wikipedia and on:
    # x^2 = 1 / 2 (T0 + T2)
    M = zeros(Float64, (N+2, N))
    @views M[1, [1, 3]] .= (1. / 2, 1. / 4)
    @views M[2, [2, 4]] .= (3. / 4,  1. / 4)
    @views M[3, [1, 3, 5]] .= (1. / 2, 1. / 2, 1. / 4) 
    for j̃ in 4:N
        M[j̃, j̃ - 2] = 1. / 4
        M[j̃, j̃] = 1. / 2
        if j̃ <= N -2
            M[j̃,j̃ + 2] = 1. / 4
        end
    end
    M[N + 1, N - 1] = 1. / 4
    M[N + 2, N] = 1. / 4
    return M
end

function multiplication_weights(N:: Int):: Vector{ComplexF64}
    V = Vector{ComplexF64}(undef, N)
    @inbounds for ĩ in 1:N
        i = ĩ - 1
        l = i ÷ 2
        if iseven(i)
            V[ĩ] = ((-1) ^ l) * π 
        else
            V[ĩ] = -im * ((-1) ^ l) * π 
        end
    end
    return V
end

struct First3MomentsChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs0_buffer:: Vector{Float64}
    coeffs1_buffer:: Vector{Float64}
    coeffs2_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}
    M_1:: Matrix{Float64}
    M_2:: Matrix{Float64}
    multiplication_weights:: Vector{ComplexF64}
    multiplication_buffer:: Vector{ComplexF64}

    function First3MomentsChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs0_buffer = Vector{Float64}(undef, N)
        coeffs1_buffer = Vector{Float64}(undef, N + 1)
        coeffs2_buffer = Vector{Float64}(undef, N + 2)
        bessels_buffer = Vector{Float64}(undef, N + 2)
        M_1 = mul_x(N)
        M_2 = mul_x_squared(N)
        weights = multiplication_weights(N + 2)
        multiplication_buffer = Vector{ComplexF64}(undef, N + 2)
        return new{N}(points, coeffs0_buffer, coeffs1_buffer, coeffs2_buffer, 
                      bessels_buffer, M_1, M_2, weights, multiplication_buffer)
    end
end

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: First3MomentsChebyshevPlan{N}):: AbstractVector{Float64} where N
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in enumerate(chebyshev_plan.points)
        chebyshev_plan.coeffs0_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    chebyshev_plan.coeffs0_buffer
end

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: First3MomentsChebyshevPlan{N}) where N
    chebyshevtransform!(values!(f, a, b, chebyshev_plan), Val(1))
    mul!(chebyshev_plan.coeffs1_buffer, chebyshev_plan.M_1, chebyshev_plan.coeffs0_buffer)
    mul!(chebyshev_plan.coeffs2_buffer, chebyshev_plan.M_2, chebyshev_plan.coeffs0_buffer);
end

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: Tuple{ComplexF64, ComplexF64, ComplexF64} where N
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N+1), k̃)
    e = cis(-k * translation) * scale
    @. chebyshev_plan.multiplication_buffer = e * chebyshev_plan.bessels_buffer * chebyshev_plan.multiplication_weights
    m0 = chebyshev_plan.coeffs0_buffer ⋅ (@views chebyshev_plan.multiplication_buffer[1:end-2])
    m1 = chebyshev_plan.coeffs1_buffer ⋅ (@views chebyshev_plan.multiplication_buffer[1:end-1])
    m2 = chebyshev_plan.coeffs2_buffer ⋅ chebyshev_plan.multiplication_buffer
    return m0, translation * m0 + scale * m1, (translation ^ 2) * m0 + 2 * translation * scale * m1 + (scale ^ 2) * m2 
end

end