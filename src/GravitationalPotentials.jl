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

function ψ_source(ks:: Vector{Vec3}, 
                  snapshot:: BubblesSnapShot, 
                  t:: Float64;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1., 
                  G:: Float64 = 1., 
                  krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                  kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    bubbles = current_bubbles(snapshot, t)
    return (4π * a^2 * G) .* k̂ik̂jTij(ks, bubbles; krotations=krotations, ΔV=ΔV)
end

export ψ

function ψ(ks:: Vector{Vec3}, 
           snapshot:: BubblesSnapShot, 
           t:: Float64;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1., 
           G:: Float64 = 1., 
           krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
           kwargs...):: Vector{ComplexF64}
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, snapshot, τ;
                                                   ΔV=ΔV, a=a, G=G, 
                                                   krotations=krotations, 
                                                   kwargs...) * (t - τ)
    return quadgk(f, 0., t; kwargs...)[1]
end

export ψ

function Ŋ(ks:: Vector{Vec3}, 
           bubbles:: Bubbles;
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    c = @. (-12π * G * a ^ 2) / (ks ⋅ ks)
    return @. c * $ŋ_source(ks, bubbles; ΔV=ΔV, kwargs...)
end

export Ŋ

function Φ(ŋ:: Vector{ComplexF64}, Ψ:: Vector{ComplexF64}):: Vector{ComplexF64}
    return ŋ - Ψ
end

function ΦminusΨ(ks:: Vector{Vec3}, 
                 snapshot:: BubblesSnapShot, 
                 t:: Float64;
                 ΔV:: Float64 = 1., 
                 a:: Float64 = 1., 
                 G:: Float64 = 1., 
                 krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                 kwargs...)
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    _ψ = ψ(ks, snapshot, t; ΔV=ΔV, a=a, G=G, krotations=krotations, 
           kwargs...)
    _ŋ = Ŋ(ks, current_bubbles(snapshot, t); ΔV=ΔV, a=a, G=G, kwargs...)
    return @. _ŋ - 2 * _ψ
end

end