using Bessels
import Bessels: besselj!
import EnvelopeApproximation.ChebyshevCFT: scale, translation, multiplication_weights, chebyshevpoints
using FastTransforms


struct SymmetricTensorChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Matrix{Float64}
    bessels_buffer:: Vector{Float64}
    multiplication_weights:: Matrix{ComplexF64}
    multiplication_buffer:: Matrix{ComplexF64}
    transform_plan!:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}

    function SymmetricTensorChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N, 6)
        bessels_buffer = Vector{Float64}(undef, N)
        weights = reshape(multiplication_weights(N), N, 1)
        multiplication_buffer = Matrix{ComplexF64}(undef, N, 6)
        transform_plan! = plan_chebyshevtransform!(zeros(N), Val(1))
        return new{N}(points, coeffs_buffer,  
                      bessels_buffer, weights, multiplication_buffer, transform_plan!)
    end
end

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: SymmetricTensorChebyshevPlan{N}):: AbstractVector{Float64} where N
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in enumerate(chebyshev_plan.points)
        @views @. chebyshev_plan.coeffs_buffer[i, :] = $f($inverse_u(u, scale_factor, t)) * $inverse_chebyshev_weight(u) 
    end     
end

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: SymmetricTensorChebyshevPlan{N}) where N
    values!(f, a, b, chebyshev_plan)
    for i in 1:6
        chebyshev_plan.transform_plan * (@views chebyshev_plan.coeffs_buffer[:, i])
    end 
end

export fourier_mode

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: SymmetricTensorChebyshevPlan{N}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: Tuple{ComplexF64, ComplexF64, ComplexF64} where N
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N+1), k̃)
    e = cis(-k * translation) * scale
    @. chebyshev_plan.multiplication_buffer = (e * chebyshev_plan.bessels_buffer * chebyshev_plan.multiplication_weights) * chebyshev_plan.coeffs_buffer
    return sum(chebyshev_plan.multiplication_buffer, dims=1)
end