begin
    using EnvelopeApproximation
    using EnvelopeApproximation.GeometricStressEnergyTensor: ring_dome_complement_intersection, IntersectionDome, Vec3, EmptyArc, FullCircle
    using EnvelopeApproximation.GeometricStressEnergyTensor: PeriodicInterval, a, b
    using Test
    using Intervals
    import Intervals.IntervalSet
    import Base.≈
    using StableRNGs
    using Random
    import Base.rand
    using Distributions
end

≈(p1:: PeriodicInterval, p2:: PeriodicInterval) = (p1.ϕ1 ≈ p2.ϕ1) & (p1.Δ ≈ p2.Δ)

@testset "ring_dome_complement_intersection simple example benchmark" begin

μ_p, μ_m = 1. / sqrt(8) - √(6) / 4, 1. / sqrt(8) + √(6) / 4
n̂ = Vec3(1., 0., 1.) / √(2)
h = 0.5
R = 1.
μs = -1.:0.001:1.

function analytic_intersection(μ)
    if μ ∈ (-1. .. μ_p)
        return EmptyArc
    elseif μ ∈ (μ_p .. μ_m)
        x = 1. / sqrt(2) - μ
        y_p = sqrt(1. / 2 + sqrt(2) * μ - 2 * (μ ^ 2))
        y_m = -y_p
        ϕ_p = atan(y_p, x)
        ϕ_m = atan(y_m, x)
        return PeriodicInterval(ϕ_m, ϕ_p - ϕ_m)
    elseif μ ∈ (μ_m .. 1.)
        return FullCircle
    end 
end

function numeric_intersection(μ)
    return ring_dome_complement_intersection(μ, R, n̂, h, true)
end

function numeric_not_domelike_intersection(μ)
    return ring_dome_complement_intersection(μ, R, n̂, h, false)
end

complement(p:: PeriodicInterval) = PeriodicInterval(mod(p.ϕ1 + p.Δ, 2π), 2π - p.Δ)

@test all(complement.(analytic_intersection.(μs)) .≈ numeric_intersection.(μs))
@test all(numeric_not_domelike_intersection.(μs) .≈ complement.(numeric_intersection.(μs)))
end;

