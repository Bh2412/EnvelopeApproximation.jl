module GravitationalWaves

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.CFTInterface: CFTPlan, fourier_modes
using EnvelopeApproximation.AngularIntegrationInterface: AbstractAngularIntegrationPlan, integrate_angular
using Random

export Π, Π_single, AzimuthalReduction, MonteCarloSampling

include("GravitationalWaves/AzimuthalReductionBigpi.jl")
include("GravitationalWaves/MonteCarloBigpi.jl")

# Inside GravitationalWaves.jl
abstract type AbstractPiStrategy end

# Strategy 1: The reduction method
struct AzimuthalReduction <: AbstractPiStrategy
    plan::CFTPlan
    angular_plan::AbstractAngularIntegrationPlan
    buffer::Union{AzimuthalReductionBigpi.x̂_ix̂_j, AzimuthalReductionBigpi.Λx̂x̂}
end

# Strategy 2: The Monte Carlo method
struct MonteCarloSampling <: AbstractPiStrategy
    N_samples::Int
    rng::AbstractRNG
end

MonteCarloSampling(N::Int) = MonteCarloSampling(N, Random.default_rng())
MonteCarloSampling(;N:: Int, rng:: Random.AbstractRNG=Random.default_rng) = MonteCarloSampling(N, rng)

function Π(t1:: Real, t2:: Real, ωs:: AbstractVector{<: Real}, snapshot:: BubblesSnapShot, space:: AbstractSpace, 
           boundary_condition:: BoundaryCondition, strategy::AbstractPiStrategy; ΔV=1.0)
    throw("Not Implemented for $(typeof(strategy))")
end

function Π(t1:: Real, t2:: Real, ωs:: AbstractVector{<: Real}, snapshot:: BubblesSnapShot, space:: AbstractSpace, 
           boundary_condition:: BoundaryCondition, strategy::AzimuthalReduction; ΔV=1.0)
    # Calls the logic from AzimuthalReductionBigpi.jl [cite: 61, 66]
    return AzimuthalReductionBigpi.Π(t1, t2, ωs, snapshot, space, boundary_condition, 
                                     strategy.plan, strategy.angular_plan, strategy.buffer; ΔV=ΔV)
end

function Π(t1:: Real, t2:: Real, ωs:: AbstractVector{<: Real}, snapshot:: BubblesSnapShot, space:: AbstractSpace,
           boundary_condition:: BoundaryCondition, strategy::MonteCarloSampling; ΔV=1.0)
    # Calls the logic from MonteCarloBigpi.jl
    # Note: MC method currently assumes BoxSpace/Periodic logic inside
    return MonteCarloBigpi.Π(t1, t2, ωs, snapshot, space, boundary_condition;
                             N_samples=strategy.N_samples, rng=strategy.rng, ΔV=ΔV)
end

function Π_single(t1::Real, t2::Real, ks::AbstractVector{<:Real}, snapshot::BubblesSnapShot, space::AbstractSpace,
                   boundary_condition::BoundaryCondition, strategy::MonteCarloSampling; ΔV=1.0)
    return MonteCarloBigpi.Π_single(t1, t2, ks, snapshot, space, boundary_condition;
                                    N_samples=strategy.N_samples, rng=strategy.rng, ΔV=ΔV)
end

end