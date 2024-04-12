module GravitationalPotentials

module SecondOrderODESolver

struct Source
    times:: Vector{Float64}
    values:: Vector{Float64}
end

export Source

pulse_times(times:: Vector{Float64}):: Vector{Float64} = (times[2:end] + times[1:end-1]) / 2

function pulse_response_matrix(times:: Vector{Float64}):: Matrix{Float64}
    n = length(times)
    times = reshape(times, (n, 1))
    earlier_times = times[1:end-1] |> x -> reshape(x, (1, n - 1))
    later_times = times[2:end] |> x -> reshape(x, (1, n - 1))
    mat = ((1 / 2) * (later_times - earlier_times)) .* (2 * times .- (earlier_times + later_times))
    mat[times .< later_times] .= 0.
    return mat
end

source_values(source:: Source):: Vector{Float64} = (source.values[1:end-1] + source.values[2:end]) / 2

function ode_solution(source:: Source):: Vector{Float64}
    pr_matrix = pulse_response_matrix(source.times)
    return pr_matrix * source_values(source)
end

export ode_solution

end
end