using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using StableRNGs
using Base.Iterators
using Measurements
using JLD2
using Plots
using EnvelopeApproximation.Visualization
import EnvelopeApproximation.BubblesEvolution.sample
using BenchmarkTools

#=
Parameters taken from Kosowsky and Turner
=#

β = 1.
Δt = (1 / β) / 100
# Γ_0 here is an integral over all space, whereas it is 
ball_space = BallSpace(4.46 / β, Point3(0., 0., 0.))
ball_space_volume = (4π / 3) * ball_space.radius ^ 3
eg = ExponentialGrowth(β, Δt, Γ_0 = ball_space_volume * 1.38 * 1e-3 * β ^ 4)
ensemble_size = 1000

N = 100
ηs = 0.5: 0.05: 0.95
rng = StableRNG(1)

function termination_strategy(η:: Float64):: Function
function ts(state, space, _):: Bool
    ps = sample(rng, N, space)
    cbs = current_bubbles(state)
    length(cbs) == 0. && return false
    inside = sum((p ∈ cbs for p in ps))
    return inside / N ≥ η 
end
    return ts
end

rngs = StableRNG.(1:ensemble_size)
@btime evolve($eg, $ball_space, termination_strategy=$termination_strategy(ηs[1]), rng=$rngs[1])
evolves = [evolve(eg, ball_space, termination_strategy=termination_strategy(η), rng=rng) for rng in rngs for η in ηs]
evolves = reshape(evolves, length(ηs), ensemble_size)

## Mean number of nucleations

mean(x) = sum(x) / length(x)
std(x) = sqrt(sum(x .^ 2) .- mean(x) ^ 2) / (sqrt(length(x) * (length(x) - 1)))
best_estimate(x) = mean(x) ± std(x) 

# mean number of nucleations
nucleations(x) = length(x.nucleations)
nucleations_sample = @. best_estimate($eachrow(nucleations(evolves)))
scatter(ηs, nucleations_sample, xscale=:log10, xlabel="η", ylabel="N")


# mean time of PT as defined by time of last nucleation
completion_time(x) = x.nucleations[end].time
completion_times_sample =  @. best_estimate($eachrow(completion_time(evolves)))
scatter(ηs, completion_times_sample, xscale=:log10, xlabel="η", ylabel="T[1/β]")

jldsave("./termination_fraction_dependence_ensemble.jld2"; snapshots=evolves, β=β)