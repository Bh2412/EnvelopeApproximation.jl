using Intervals
using FastTransforms
using SpecialFunctions
using QuadGK
using ApproxFun
using BenchmarkTools
using Plots

const ChebyshevInterval:: Intervals.Interval = Intervals.Interval(-1. ,1.)

translation(a:: Float64, b:: Float64):: Float64 = (a + b) / 2
scale(a:: Float64, b:: Float64):: Float64 = (b - a) / 2
u(x:: Float64, scale:: Float64, translation:: Float64):: Float64 = (x - translation) / scale
inverse_u(u:: Float64, scale:: Float64, translation:: Float64):: Float64 = translation + scale * u
inverse_chebyshev_weight(u:: Float64):: Float64 = sqrt(1 - u ^ 2)


function values!(f, a:: Float64, b:: Float64, n:: Int, 
                 V:: Vector{Float64}):: AbstractVector{Float64}
    us = chebyshevpoints(Float64, n, Val(1))
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in zip(1:n, us)
        V[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    @views return V[1:n]
end

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, n:: Int,
                           V:: Vector{Float64}):: Vector{Float64}
    Ṽ = values!(f, a, b, n, V)
    return chebyshevtransform!(Ṽ, Val(1))
end

# Followins equation 2.10.5.2 in "methods of numerical integration"

function cosine_mode(k:: Float64, chebyshev_coeffs:: Vector{Float64})
    return sum(chebyshev_coeffs[2i + 1] * ((-1) ^ i) * π * besselj(2i, k)
               for i in 0:1:((length(chebyshev_coeffs) - 1) ÷ 2))
end

# Followins equation 2.10.5.1 in "methods of numerical integration"

function sine_mode(k:: Float64, chebyshev_coeffs:: Vector{Float64})
    return sum(chebyshev_coeffs[2i + 2] * ((-1) ^ i) * π * besselj(2i + 1 , k)
               for i in 0:1:((length(chebyshev_coeffs) - 2) ÷ 2))
end

function fourier_mode(k:: Float64, 
                      chebyshev_coeffs:: Vector{Float64}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: ComplexF64
    k̃ = scale * k
    v = cosine_mode(k̃, chebyshev_coeffs) - im * sine_mode(k̃, chebyshev_coeffs)    
    return v * cis(-k * translation) * scale
end

function _fourier_mode(k:: Float64, 
                       chebyshev_coeffs:: Vector{Float64}, 
                       scale:: Float64 = 1.,
                       translation:: Float64 = 0.):: ComplexF64
    k̃ = scale * k
    besselj_vals = Bessels.besselj(0:length(chebyshev_coeffs), k̃)
    c =sum(chebyshev_coeffs[2i + 1] * ((-1) ^ i) * π * besselj_vals[2i + 1]
           for i in 0:1:((length(chebyshev_coeffs) - 1) ÷ 2))
    s = sum(chebyshev_coeffs[2i + 2] * ((-1) ^ i) * π * besselj_vals[2i + 2]
            for i in 0:1:((length(chebyshev_coeffs) - 2) ÷ 2))
    return (c - im * s) * cis(-k * translation) * scale
end

function fourier_modes(ks:: AbstractVector{Float64}, 
                       f, a:: Float64, b:: Float64, n:: Int)
    V = Vector{Float64}(undef, n)
    coeffs = chebyshev_coeffs!(f, a, b, n, V)
    t = translation(a, b)
    a = scale(a, b)
    return fourier_mode.(ks, (coeffs, ), (a, ), (t, ))
end

function fourier(g, k:: Float64, a:: Float64, b:: Float64; 
                 kwargs...)
    return quadgk(x -> g(x) * cis(-k * x), a, b; kwargs...)
end

V = zeros(2^10)
n = 2^4
a, b = -1., 1.
exp_coeffs = chebyshev_coeffs!(exp, a, b, n, V)
exp_approximation = x -> [cos(k*acos(x)) for k=0:n-1]' * exp_coeffs
x = a:0.01:b
plot(x, [exp(z) * sqrt(1 - z ^2) for z in x], label = "exp(x)")
plot!(x, exp_approximation.(x), label = "exp(x) approximation")

# Example

a, b = 1., 2.
n = 2 ^ 2
ks = 0.01:0.01:10
new_method_modes = fourier_modes(ks, sin, a, b, n)
quadgk_modes = fourier.((sin, ), ks, (a, ), (b, )) .|>  x -> x[1]

(new_method_modes - quadgk_modes) .|> abs |> x -> max(x...)


sin_coeffs = chebyshev_coeffs!(sin, a, b, n, V)
# @btime sin_coeffs = chebyshev_coeffs!(sin, a, b, n, V)
scale_factor = scale(a, b)
t = translation(a, b)
k = ks[end]
@btime fourier_mode($k, $sin_coeffs, $scale_factor, $t)
@btime _fourier_mode($k, $sin_coeffs, $scale_factor, $t)
@profview for _ in 1:100_000 _fourier_mode(k, sin_coeffs, scale_factor, t) end

function bessel_table(max_order:: Int, ks:: Vector{Float64}):: Matrix{Float64}
    table = Matrix{Float64}(undef, max_order, length(ks))
    for (j, k) in enumerate(ks)
        for i in 1:max_order
            table[i, j] = besselj(i-1, k)
        end
    end
    return table
end


function cosine_mode(precomputed_bessels:: Vector{Float64}, chebyshev_coeffs:: Vector{Float64})
    return sum(chebyshev_coeffs[2i + 1] * ((-1) ^ i) * π * precomputed_bessels[2i + 1]
               for i in 0:1:((length(chebyshev_coeffs) - 1) ÷ 2))
end

# Followins equation 2.10.5.1 in "methods of numerical integration"

function sine_mode(precomputed_bessels:: Vector{Float64}, chebyshev_coeffs:: Vector{Float64})
    return sum(chebyshev_coeffs[2i + 2] * ((-1) ^ i) * π * precomputed_bessels[2i + 2]
               for i in 0:1:((length(chebyshev_coeffs) - 2) ÷ 2))
end

function fourier_mode(k:: Float64,
                      chebyshev_coeffs:: Vector{Float64}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: ComplexF64
    k̃ = scale * k
    v = cosine_mode(k̃, chebyshev_coeffs) - im * sine_mode(k̃, chebyshev_coeffs)    
    return v * cis(-k * translation) * scale
end

