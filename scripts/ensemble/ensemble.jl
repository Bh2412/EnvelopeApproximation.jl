begin
    using EnvelopeApproximation
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    using StableRNGs
    using Base.Iterators
    using Measurements
    using JLD2
    using EnvelopeApproximation.Visualization
    import EnvelopeApproximation.BubblesEvolution.sample
    using BenchmarkTools
end

#=
Parameters taken from Kosowsky and Turner
=#

begin
β = 1.
Δt = (1 / β) / 1000
ball_space = BallSpace(4.46 / β, Point3(0., 0., 0.))
ball_space_volume = 4π / 3 * ball_space.radius ^ 3
# Γ(t) in Kosowsky and Turner differ from this work by a factor of unoccupied volume
eg = ExponentialGrowth(β, Δt, Γ_0 = ball_space_volume * 1.38 * 1e-3 * β ^ 4)
ensemble_size = 1000

N = 100
η = 0.95
rng = StableRNG(1)

points_buffer = Vector{Point3}(undef, N)
bubbles_buffer = Vector{Bubble}(undef, 10_000)
function termination_strategy(state, space, _):: Bool
    ps = sample!(rng, N, space, points_buffer)
    cbs = current_bubbles!(state, bubbles_buffer)
    length(cbs) == 0 && return false
    inside = sum((p ∈ cbs for p in ps), init=0.)
    return inside / N ≥ η 
end

_evolve(rng) = evolve(eg, ball_space, termination_strategy=termination_strategy, rng=rng)
end

begin
rngs = StableRNG.(1:ensemble_size)
evolves = _evolve.(rngs)
jldsave("evolution_ensemble.jld2"; snapshots=evolves, β=β, ball_space=ball_space)
end
