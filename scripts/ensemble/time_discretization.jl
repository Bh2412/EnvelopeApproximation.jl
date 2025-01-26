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
ENV["JULIA_DEBUG"] = EnvelopeApproximation

#=
Parameters taken from Kosowsky and Turner
=#

β = 1.
Δt = β / 100
bs = BallSpace(4.46 / β, Point3(0., 0., 0.))
bs_volume = bs.radius ^ 3 * (4π / 3)
eg = ExponentialGrowth(β, Δt, Γ_0 = bs_volume * 1.38 * 1e-3 * β ^ 4)


N = 100
η = 0.8
rng = StableRNG(1)
function ts(state, space, _):: Bool
    ps = sample(rng, N, space)
    cbs = current_bubbles(state)
    length(cbs) == 0. && return false
    inside = sum([p ∈ cbs for p in ps])
    return inside / N ≥ η 
end

Δts = 10 .^ (range(-3., 0., 20))
egs = ExponentialGrowth.(β, Δts, Γ_0 = 1.) 
rngs = StableRNG.(1:30)

_evolve(eg, rng) = evolve(eg, bs, termination_strategy=ts, rng=rng)
_evolve(t:: Tuple) = _evolve(t...)

evolves = _evolve.(product(egs, rngs))

## Mean number of nucleations

mean(x) = sum(x) / length(x)
std(x) = sqrt(sum(x .^ 2) .- mean(x) ^ 2) / (sqrt(length(x) * (length(x) - 1)))
average(x) = mean(x) ± std(x) 
mean_nucleations(x) = average((z -> length(z.nucleations)).(x))

nucs = mean_nucleations.(eachrow(evolves))
custom_xticks = 10. .^(-3, -2, -1, 0)
custom_xticks_strs = ["$f" for f in custom_xticks]
scatter(Δts, nucs, xscale=:log10, xticks=(custom_xticks, custom_xticks_strs), xlabel="Δt [β]", ylabel="N")

jldsave("./time_discretization_ensemble.jld2"; snapshots=evolves, β=β)