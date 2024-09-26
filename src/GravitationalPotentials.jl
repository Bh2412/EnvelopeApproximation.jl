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
using EnvelopeApproximation.StressEnergyTensor
import EnvelopeApproximation.BubbleBasics: Point3, coordinates, Vec3
import EnvelopeApproximation.StressEnergyTensor: upper_right, diagonal, above_diagonal
using HCubature
import LinearAlgebra: norm

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

function v̂iv̂j(ks:: Vector{Vec3}, tds:: Vector):: Array{Float64, 2}
    return @. v̂iv̂j($reshape(ks, :, 1), $reshape(tds, 1, :))
end

function δij(td:: Tuple{Symbol, Symbol}):: Float64
    (td[1] ≡ td[2]) && return 1.
    return 0.
end

function ψ_source(ks:: Vector{Vec3}, 
                  T:: Matrix{ComplexF64}, 
                  tensor_directions:: Vector, 
                  a = 1., 
                  G = 1.):: Vector{ComplexF64}
    ```math
    ψ'' = 4πG_N a^2 k̂ik̂j Tij
    ```        
    k_ik_j = v̂iv̂j(ks, tensor_directions)
    V = zeros(ComplexF64, length(ks))
    for (i, td) in enumerate(tensor_directions)
        if td in above_diagonal
            @. V += (2 * k_ik_j[:, i]) * T[:, i]
        elseif td in diagonal
            @. V += k_ik_j[:, i] * T[:, i] 
        end
    end
    return (4π * a^2 * G) * V  
end

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  t:: Float64, 
                  ϕ_resolution:: Float64,
                  μ_resolution:: Float64,
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1.; 
                  kwargs...):: Vector{ComplexF64}
    tensor_directions = Vector{TensorDirection}(upper_right)
    bubbles = current_bubbles(snapshot, t)
    T = T_ij(ks, bubbles, ϕ_resolution, μ_resolution, ΔV, tensor_directions; kwargs...)
    return ψ_source(ks, T, tensor_directions, a, G)
end

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  t:: Float64, 
                  n_ϕ:: Int,
                  n_μ:: Int,
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1.; 
                  kwargs...):: Vector{ComplexF64}
    return ψ_source(ks, snapshot, t, 2π / n_ϕ, 2. / n_μ, ΔV, a, G; kwargs...)
end

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  times:: Vector{Float64}, 
                  ϕ_resolution:: Float64,
                  μ_resolution:: Float64,
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1.; 
                  kwargs...):: Matrix{ComplexF64}
    tensor_directions = Vector{TensorDirection}(upper_right)
    S = zeros(ComplexF64, length(times), length(ks))
    for (i, t) in enumerate(times)
        bubbles = current_bubbles(snapshot, t)
        T = T_ij(ks, bubbles, ϕ_resolution, μ_resolution, ΔV, tensor_directions; kwargs...)
        S[i, :] .= ψ_source(ks, T, tensor_directions, a, G)
    end
    return S
end

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  times:: Vector{Float64}, 
                  n_ϕ:: Int64,
                  n_μ:: Int64,
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1.; 
                  kwargs...):: Matrix{ComplexF64}
    return ψ_source(ks, snapshot, times, 2π / n_ϕ, 2. / n_μ, 
                    ΔV, a, G; kwargs...)
end

export quad_ψ

function quad_ψ(ks:: Vector{Vec3}, 
                snapshot:: BubblesSnapShot, 
                t:: Float64, 
                ϕ_resolution:: Float64,
                μ_resolution:: Float64,
                ΔV:: Float64 = 1., 
                a:: Float64 = 1., 
                G:: Float64 = 1.; 
                kwargs...):: Vector{ComplexF64}
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, snapshot, τ, ϕ_resolution, μ_resolution, 
                                                   ΔV, a, G; kwargs...) * (t - τ)
    return hquadrature(f, 0., t; kwargs...)[1]
end

function quad_ψ(ks:: Vector{Vec3}, 
                snapshot:: BubblesSnapShot, 
                t:: Float64, 
                n_ϕ:: Int64,
                n_μ:: Int64,
                ΔV:: Float64 = 1., 
                a:: Float64 = 1., 
                G:: Float64 = 1.; 
                kwargs...):: Vector{ComplexF64}
    return quad_ψ(ks, snapshot, t, 2π / n_ϕ, 2. / n_μ, ΔV, a, G; kwargs...)
end

function ψ(times:: Vector{Float64}, 
           S:: Matrix{ComplexF64}):: Matrix{ComplexF64}
    source = Source(times, S)
    return ode_solution(source)
end

function ψ(ks:: Vector{Vec3}, 
           snapshot:: BubblesSnapShot, 
           times:: Vector{Float64}, 
           ϕ_resolution:: Float64, 
           μ_resolution:: Float64,
           ΔV:: Float64 = 1., 
           a:: Float64 = 1., 
           G:: Float64 = 1.; 
           kwargs...)
    S = ψ_source(ks, snapshot, times, ϕ_resolution, μ_resolution, 
                 ΔV, a, G; kwargs...)
    return ψ(times, S)
end

function ψ(ks:: Vector{Vec3}, 
           snapshot:: BubblesSnapShot, 
           times:: Vector{Float64}, 
           n_ϕ:: Int64, 
           n_μ:: Int64,
           ΔV:: Float64 = 1., 
           a:: Float64 = 1., 
           G:: Float64 = 1.; 
           kwargs...)
    return ψ(ks, snapshot, times, 2π / n_ϕ, 2. / n_μ, 
             ΔV, a, G)
end

export ψ

function Ŋ(ks:: Vector{Vec3},
           T:: Matrix{ComplexF64},
           tensor_directions:: Vector,
           a:: Float64,
           G:: Float64
           ):: Vector{ComplexF64}
    c = @. (-12π * G * a ^ 2) / (ks ⋅ ks) 
    A = @. $v̂iv̂j(ks, tensor_directions) - ($reshape(δij(tensor_directions), 1, :) / 3)
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
    T = T_ij(ks, bubbles, ϕ_resolution, μ_resolution, ΔV, tensor_directions; kwargs...)
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