module GravitationalPotentials

module SecondOrderODESolver

import Tullio: @tullio

struct Source{T}
    times:: Vector{Float64}
    values:: T
end

export Source

pulse_times(times:: Vector{Float64}):: Vector{Float64} = (times[2:end] + times[1:end-1]) / 2

function pulse_response_matrix(times:: Vector{Float64}):: Matrix{Float64}
    n = length(times)
    times = reshape(times, (n, 1))
    earlier_times = times[1:end-1] |> x -> reshape(x, (1, n - 1))
    later_times = times[2:end] |> x -> reshape(x, (1, n - 1))
    mat = ((1 / 2) * (later_times - earlier_times)) .* (2 * times .- (earlier_times + later_times))
    # The response to source pulses that arrive later is null.
    mat[times .< later_times] .= 0.
    return mat
end

function source_values(source:: Source{Vector{Float64}}):: Vector{Float64}
    return (source.values[1:end-1] + source.values[2:end]) / 2
end

function source_values(source:: Source{T}):: T where T <: Matrix
    return (source.values[1:end-1, :] + source.values[2:end, :]) / 2
end

function source_values(source:: Source{Array{Float64, 3}}):: Array{Float64, 3}
    return (source.values[1:end-1, :, :] + source.values[2:end, :, :]) / 2
end

function ode_solution(source:: Source{Vector{Float64}}):: Vector{Float64}
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    return pr_matrix * sv
end

function ode_solution(source:: Source{T}):: T where T <: Matrix
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    return pr_matrix * sv
end

function ode_solution(source:: Source{Array{Float64, 3}}):: Array{Float64, 3}
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    @tullio M[i, j, k] := pr_matrix[i, l] * sv[l, j, k]
    return M
end


export ode_solution

end
 
using EnvelopeApproximation.GravitationalPotentials.SecondOrderODESolver
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.BubbleBasics: Point3, coordinates, Vec3
using QuadGK
using StaticArrays
import LinearAlgebra: norm

# In accordange with the T_ij function
const TensorDirections:: Vector{Tuple{Symbol, Symbol}} = [(:x, :x), (:x, :y), (:x, :z), (:y, :y), (:y, :z), (:z, :z)]
const AboveDiagonal:: Vector{Tuple{Symbol, Symbol}} = [(:x, :y), (:x, :z), (:y, :z)]
const Diagonal:: Vector{Tuple{Symbol, Symbol}} = [(:x, :x), (:y, :y), (:z, :z)]

function v̂(v:: Vec3):: Vec3
    return v ./ norm(v)
end

function v̂iv̂j(k:: Vec3, td:: Tuple{Symbol, Symbol})
    ```math
    k_ik_j / k^2
    ```        
    indices = indexin(td, [:x, :y, :z])
    return prod(v̂(k)[indices])
end

function v̂iv̂j(ks:: Vector{Vec3}, tds:: Vector{Tuple{Symbol, Symbol}} = TensorDirections):: Array{Float64, 2}
    return @. v̂iv̂j($reshape(ks, :, 1), $reshape(tds, 1, :))
end

function δij(td:: Tuple{Symbol, Symbol}):: Float64
    (td[1] ≡ td[2]) && return 1.
    return 0.
end

function ψ_source(ks:: Vector{Vec3}, 
                  T:: Matrix{ComplexF64}, 
                  a = 1., 
                  G = 1.):: Vector{ComplexF64}
    ```math
    ψ'' = 4πG_N a^2 k̂ik̂j Tij
    ```        
    k_ik_j = v̂iv̂j(ks, TensorDirections)
    V = zeros(ComplexF64, length(ks))
    for (i, td) in enumerate(TensorDirections)
        if td in AboveDiagonal
            @. V += (2 * k_ik_j[:, i]) * T[:, i]
        elseif td in Diagonal
            @. V += k_ik_j[:, i] * T[:, i] 
        end
    end
    return (4π * a^2 * G) * V  
end

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  t:: Float64;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1., 
                  krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                  kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing,
                  kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    bubbles = current_bubbles(snapshot, t)
    T = T_ij(ks, bubbles; krotations=krotations, 
             kdrotations=kdrotations, ΔV=ΔV, kwargs...)
    return ψ_source(ks, T, a, G)
end


function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  times:: Vector{Float64};
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1.,
                  kwargs...):: Matrix{ComplexF64}
    S = zeros(ComplexF64, length(times), length(ks))
    for (i, t) in enumerate(times)
        bubbles = current_bubbles(snapshot, t)
        T = T_ij(ks, bubbles; ΔV=ΔV, kwargs...)
        S[i, :] .= ψ_source(ks, T, a, G)
    end
    return S
end

export ψ

function ψ(ks:: Vector{Vec3}, 
           snapshot:: BubblesSnapShot, 
           t:: Float64;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1., 
           G:: Float64 = 1., 
           krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
           kdrotations:: Union{Nothing, Vector{<: SMatrix{6, 6, Float64}}} = nothing,
           kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    kdrotations ≡ nothing && (kdrotations = symmetric_tensor_inverse_rotation.(krotations))
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, snapshot, τ, 
                                                   ΔV=ΔV, a=a, G=G, 
                                                   krotations=krotations, 
                                                   kdrotations=kdrotations; kwargs...) * (t - τ)
    return quadgk(f, 0., t; kwargs...)[1]
end

function Ŋ(ks:: Vector{Vec3},
           T:: Matrix{ComplexF64},
           tensor_directions:: Vector,
           a:: Float64,
           G:: Float64
           ):: Vector{ComplexF64}
    c = @. (-12π * G * a ^ 2) / (ks ⋅ ks)
    δ = reshape(δij.(tensor_directions), 1, :)
    A = @. ($v̂iv̂j(ks, tensor_directions) - (δ / 3)) * (2 - δ)  # The 2 - δ is due to the concatenation of a symmetric tensor
    return @. $sum((c * A * T), dims=2)[:]
end

export ψ

export Ŋ

function Ŋ(ks:: Vector{Vec3}, 
           bubbles:: Bubbles,
           ϕ_resolution:: Float64, 
           μ_resolution:: Float64, 
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1.; kwargs...):: Vector{ComplexF64}
    tensor_directions = Vector{TensorDirection}(upper_right)
    T = T_ij(ks, bubbles; ΔV=ΔV, kwargs...)
    return Ŋ(ks, T, tensor_directions, a, G)
end

function Ŋ(ks:: Vector{Vec3}, 
           bubbles:: Bubbles,
           n_ϕ:: Int64, 
           n_μ:: Int64, 
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1.; kwargs...):: Vector{ComplexF64}
    return Ŋ(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV, a, G; kwargs...)
end

function Φ(ŋ:: Vector{ComplexF64}, Ψ:: Vector{ComplexF64}):: Vector{ComplexF64}
    return ŋ - Ψ
end

end