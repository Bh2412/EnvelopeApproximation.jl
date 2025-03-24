module ChebyshevCFT

using Bessels
import Bessels: besselj!
using LinearAlgebra
using FastTransforms
using StaticArrays

# Following chapter 2.10.5 in "methods of numerical integration"

translation(a:: Float64, b:: Float64):: Float64 = (a + b) / 2
scale(a:: Float64, b:: Float64):: Float64 = (b - a) / 2
u(x:: Float64, scale:: Float64, translation:: Float64):: Float64 = (x - translation) / scale
inverse_u(u:: Float64, scale:: Float64, translation:: Float64):: Float64 = translation + scale * u
inverse_chebyshev_weight(u:: Float64):: Float64 = sqrt(1 - u ^ 2)

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

export ChebyshevPlan

struct ChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}
    multiplication_weights:: Vector{ComplexF64}
    multiplication_buffer:: Vector{ComplexF64}
    transform_plan!:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}

    function ChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N)
        bessels_buffer = Vector{Float64}(undef, N)
        weights = multiplication_weights(N)
        multiplication_buffer = Vector{ComplexF64}(undef, N)
        transform_plan! = plan_chebyshevtransform!(zeros(N), Val(1))
        return new{N}(points, coeffs_buffer,  
                      bessels_buffer, weights, multiplication_buffer, transform_plan!)
    end
end

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: ChebyshevPlan{N}) where N
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in enumerate(chebyshev_plan.points)
        chebyshev_plan.coeffs_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    return chebyshev_plan.coeffs_buffer
end

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: ChebyshevPlan{N}) where N
    chebyshev_plan.transform_plan! * values!(f, a, b, chebyshev_plan)
end

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: ChebyshevPlan{N}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: ComplexF64 where N
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N-1), k̃)
    e = cis(-k * translation) * scale
    @. chebyshev_plan.multiplication_buffer = e * chebyshev_plan.bessels_buffer * chebyshev_plan.multiplication_weights
    return chebyshev_plan.coeffs_buffer ⋅ chebyshev_plan.multiplication_buffer
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

export First3MomentsChebyshevPlan

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
    transform_plan:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}

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
        transform_plan = plan_chebyshevtransform!(zeros(N), Val(1))
        return new{N}(points, coeffs0_buffer, coeffs1_buffer, coeffs2_buffer, 
                      bessels_buffer, M_1, M_2, weights, multiplication_buffer, transform_plan)
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

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: First3MomentsChebyshevPlan{N}) where N
    chebyshev_plan.transform_plan * values!(f, a, b, chebyshev_plan)
    mul!(chebyshev_plan.coeffs1_buffer, chebyshev_plan.M_1, chebyshev_plan.coeffs0_buffer)
    mul!(chebyshev_plan.coeffs2_buffer, chebyshev_plan.M_2, chebyshev_plan.coeffs0_buffer);
end

export fourier_mode

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

export VectorChebyshevPlan

struct VectorChebyshevPlan{N, K}
    points:: Vector{Float64}
    coeffs_buffer:: Matrix{Float64}
    bessels_buffer:: Vector{Float64}
    multiplication_weights:: Vector{ComplexF64}
    multiplication_buffer:: Vector{ComplexF64}
    transform_plan!:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}
    mode_buffer:: Vector{ComplexF64}

    function VectorChebyshevPlan{N, K}() where {N, K}
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Matrix{Float64}(undef, N, K)
        bessels_buffer = Vector{Float64}(undef, N)
        weights = multiplication_weights(N)
        multiplication_buffer = Vector{ComplexF64}(undef, N)
        transform_plan! = plan_chebyshevtransform!(zeros(N), Val(1))
        mode_buffer = Vector{ComplexF64}(undef, K)
        return new{N, K}(points, coeffs_buffer,  
                         bessels_buffer, weights, multiplication_buffer, transform_plan!, mode_buffer)
    end
end

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: VectorChebyshevPlan{N, K}) where {N, K}
    scale_factor = scale(a, b)
    t = translation(a, b)
    @inbounds for (i, u) in enumerate(chebyshev_plan.points)
        icw = inverse_chebyshev_weight(u) 
        @views @. chebyshev_plan.coeffs_buffer[i, :] = $f($inverse_u(u, scale_factor, t)) * icw
    end     
end

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: VectorChebyshevPlan{N, K}) where {N, K}
    values!(f, a, b, chebyshev_plan)
    @inbounds for i in 1:K
        chebyshev_plan.transform_plan! * (@views chebyshev_plan.coeffs_buffer[:, i])
    end 
end

export fourier_mode

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: VectorChebyshevPlan{N, K}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: Vector{ComplexF64} where {N, K}
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N-1), k̃)
    e = cis(-k * translation) * scale
    @. chebyshev_plan.multiplication_buffer = e * chebyshev_plan.bessels_buffer * chebyshev_plan.multiplication_weights
    @inbounds for i in 1:K
        chebyshev_plan.mode_buffer[i] = (@views chebyshev_plan.coeffs_buffer[:, i]) ⋅ chebyshev_plan.multiplication_buffer
    end
    return chebyshev_plan.mode_buffer
end

function fourier_modes(ks:: AbstractVector{Float64}, f, a:: Real, b:: Real, 
    plan:: VectorChebyshevPlan{N, K}):: Matrix{ComplexF64} where {N, K}
    M = Matrix{ComplexF64}(undef, length(ks), K)
    chebyshev_coeffs!(f, a, b, plan)
    s = scale(a, b)
    t = translation(a, b)
    for (i, k) in enumerate(ks)
        @views M[i, :] .= fourier_mode(k, plan, s, t)
    end
    return M
end

function fourier_modes(ks:: AbstractVector{Float64}, f, a:: Real, b:: Real, 
    plan:: First3MomentsChebyshevPlan{N}):: Matrix{ComplexF64} where N
    M = Matrix{ComplexF64}(undef, length(ks), 3)
    chebyshev_coeffs!(f, a, b, plan)
    s = scale(a, b)
    t = translation(a, b)
    for (i, k) in enumerate(ks)
        @views M[i, :] .= fourier_mode(k, plan, s, t)
    end
    return M
end

export TailoredChebyshevPlan

struct TailoredChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Vector{Float64}
    multiplication_weights:: Matrix{ComplexF64}
    multiplication_buffer:: Matrix{ComplexF64}
    modes_buffer:: Matrix{ComplexF64}
    transform_plan!:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}
    ks:: Vector{Float64}
    a:: Float64
    b:: Float64

    function TailoredChebyshevPlan{N}(ks:: Vector{Float64}, 
                                      a:: Float64 = -1., 
                                      b:: Float64 = 1.) where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N)
        scale_factor = scale(a, b)
        t = translation(a, b)
        translation_factors = reshape((@. cis(-ks * t) * scale_factor), 1, :)
        _weights = reshape(multiplication_weights(N), :, 1)
        weights = @. _weights * translation_factors
        # Constructing the precomputed bessels, weighted by all other factors
        for (i, k) in enumerate(ks)
            weights[:, i] .*= besselj(0:(N-1), scale_factor * k)
        end
        multiplication_buffer = Matrix{ComplexF64}(undef, N, length(ks))
        modes_buffer = Matrix{ComplexF64}(undef, 1, length(ks))
        transform_plan! = plan_chebyshevtransform!(zeros(N), Val(1))
        return new{N}(points, coeffs_buffer,  
                      weights, multiplication_buffer, modes_buffer, transform_plan!, 
                      ks, a, b)
    end
end

scale(chebyshev_plan:: TailoredChebyshevPlan{N}) where N = scale(chebyshev_plan.a, chebyshev_plan.b)
translation(chebyshev_plan:: TailoredChebyshevPlan{N}) where N = translation(chebyshev_plan.a, chebyshev_plan.b)

function values!(f, chebyshev_plan:: TailoredChebyshevPlan{N}) where N
    scale_factor = scale(chebyshev_plan)
    t = translation(chebyshev_plan)
    for (i, u) in enumerate(chebyshev_plan.points)
        chebyshev_plan.coeffs_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    return chebyshev_plan.coeffs_buffer
end

function chebyshev_coeffs!(f, chebyshev_plan:: TailoredChebyshevPlan{N}) where N
    chebyshev_plan.transform_plan! * values!(f, chebyshev_plan)
end

function fourier_modes(chebyshev_plan:: TailoredChebyshevPlan):: Vector{ComplexF64}
    @. chebyshev_plan.multiplication_buffer = chebyshev_plan.multiplication_weights * chebyshev_plan.coeffs_buffer 
    sum!(chebyshev_plan.modes_buffer, chebyshev_plan.multiplication_buffer)
    return chebyshev_plan.modes_buffer[:]
end

end