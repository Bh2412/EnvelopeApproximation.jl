using Bessels
import Bessels: besselj!
import EnvelopeApproximation.ChebyshevCFT: values!, chebyshev_coeffs!, fourier_mode, scale, translation, multiplication_weights, chebyshevpoints, First3MomentsChebyshevPlan, inverse_chebyshev_weight, inverse_u
using FastTransforms
using BenchmarkTools
using LinearAlgebra
using Test

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
    for (i, u) in enumerate(chebyshev_plan.points)
        icw = inverse_chebyshev_weight(u) 
        @views @. chebyshev_plan.coeffs_buffer[i, :] = $f($inverse_u(u, scale_factor, t)) * icw
    end     
end

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: VectorChebyshevPlan{N, K}) where {N, K}
    values!(f, a, b, chebyshev_plan)
    for i in 1:K
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
    for i in 1:K
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


begin

vector_chebyshev_plan = VectorChebyshevPlan{32, 3}()
moments_chebyshev_plan = First3MomentsChebyshevPlan{32}()

f(x) = exp(-x^2)
vector_f(x) = begin
    e = exp(-x^2)
    return [e, e * x, e * x ^ 2]
end

ks = range(0., 10., 1_000)

vector_modes = fourier_modes(ks, vector_f, -1., 1., vector_chebyshev_plan)
moments_modes = fourier_modes(ks, f, -1., 1., moments_chebyshev_plan)

@test max(((vector_modes .- moments_modes) .|> abs)...) < 1e-13

end

begin

vector_chebyshev_plan = VectorChebyshevPlan{32, 3}()
moments_chebyshev_plan = First3MomentsChebyshevPlan{32}()

f(x) = exp(-x^2)
vector_f(x) = begin
    e = exp(-x^2)
    return [e, e * x, e * x ^ 2]
end

a, b = 0.5, 3.
ks = range(0., 10., 1_000)

vector_modes = fourier_modes(ks, vector_f, a, b, vector_chebyshev_plan)
moments_modes = fourier_modes(ks, f, a, b, moments_chebyshev_plan)

@test max(((vector_modes .- moments_modes) .|> abs)...) < 1e-13
    
end

# @btime fourier_modes($ks, $vector_f, -1., 1., $vector_chebyshev_plan)
# @btime fourier_modes($ks, $f, -1., 1., $moments_chebyshev_plan)

# @profview for _ in 1:10_000 fourier_modes(ks, vector_f, -1., 1., vector_chebyshev_plan) end
# @profview for _ in 1:10_000 fourier_modes(ks, f, -1., 1., moments_chebyshev_plan) end


