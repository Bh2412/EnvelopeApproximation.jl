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

function source_values(source:: Source{Array{Float64, 2}}):: Array{Float64, 2}
    return (source.values[1:end-1, :] + source.values[2:end, :]) / 2
end

function source_values(source:: Source{Array{Float64, 3}}):: Array{Float64, 3}
    return (source.values[1:end-1, :, :] + source.values[2:end, :, :]) / 2
end

function ode_solution(source:: Source{Vector{Float64}}):: Vector{Float64}
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    @tullio v[j] := pr_matrix[j, l] * sv[l]
    return v
end

function ode_solution(source:: Source{Array{Float64, 2}}):: Array{Float64, 2}
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    @tullio M[i, j] := pr_matrix[i, l] * sv[l, j]
    return M
end

function ode_solution(source:: Source{Array{Float64, 3}}):: Array{Float64, 3}
    pr_matrix = pulse_response_matrix(source.times)
    sv = source_values(source)
    @tullio M[i, j, k] := pr_matrix[i, l] * sv[l, j, k]
    return M
end


export ode_solution

end
end