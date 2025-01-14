module ChebyshevCFT

using Bessels

# Following chapter 2.10.5 in "methods of numerical integration"

struct ChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}

    function ChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N)
        bessels_buffer = Vector{Float64}(undef, N)
        return new{N}(points, coeffs_buffer, bessels_buffer)
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
        chebyshev_plan.coeffs_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    return chebyshev_plan.coeffs_buffer
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
    @inbounds for ((ĩ, coeff), bessel_value) in zip(enumerate(chebyshev_plan.coeffs_buffer), chebyshev_plan.bessels_buffer)
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

struct First3MomentsChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}
    M_x:: Matrix{Float64}
    M_x_squared:: Matrix{Float64}

    function First3MomentsChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N)
        bessels_buffer = Vector{Float64}(undef, N)
        M_x = mul_x(N)
        M_x_squared = mul_x_squared(N)
        return new{N}(points, coeffs_buffer, bessels_buffer, M_x, M_x_squared)
    end
end

end