begin
using EnvelopeApproximation.GeometricStressEnergyTensor: ring_dome_complement_intersection, IntersectionDome, Vec3, EmptyArc, FullCircle
using EnvelopeApproximation.GeometricStressEnergyTensor: PeriodicInterval, a, b, periodic_intersection!
using Test
import EnvelopeApproximation.GeometricStressEnergyTensor.periodic_intersection!
using Intervals
import Intervals.IntervalSet
import Base: ≈, ==
using StableRNGs
using Random
import Base.rand
using Distributions
using Combinatorics
import Base.length
end

begin

function IntervalSet(p:: PeriodicInterval)
    _a = a(p)
    _b = b(p)
    _b < _a && (return IntervalSet([0. .. _b, _a .. 2π]))
    return IntervalSet(_a .. _b)
end

function rand(rng:: AbstractRNG, ::Type{PeriodicInterval}, n:: Int)
    ϕs = rand(rng, Uniform(0., 2π), (n, 2))
    return map(x -> PeriodicInterval(x[1], x[2]), eachrow(ϕs))
end

function manual_intersect(ps:: AbstractVector{PeriodicInterval}):: IntervalSet
    return reduce(intersect, IntervalSet.(ps))
end

N = 15
random_periodic_intervals = rand(StableRNG(1), PeriodicInterval, N)
subsets = collect(powerset(random_periodic_intervals, 2, 5))
limits_buffer = Vector{Tuple{Float64, Float64}}(undef, 2N + 2)
intersection_buffer = Vector{PeriodicInterval}(undef, 2N + 2)

function Δ(intervalset:: IntervalSet)
    return sum(inter.last - inter.first for inter in intervalset.items; init=0.)
end

function Δ(ps:: AbstractVector{PeriodicInterval})
    return sum(p.Δ for p in ps; init=0.)
end

periodic_intersection!(ps:: AbstractVector{PeriodicInterval}) = periodic_intersection!(ps, limits_buffer, intersection_buffer)

bools = Vector{Bool}(undef, length(subsets))
for i in eachindex(subsets)
    subset = subsets[i]
    bools[i] = (Δ(periodic_intersection!(subset)) ≈ Δ(manual_intersect(subset)))
end
@test all(bools)
end