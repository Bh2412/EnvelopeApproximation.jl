module EnvelopeAnalysis

import EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
import EnvelopeApproximation.BubblesEvolution: BallSpace
using EnvelopeApproximation.ChebyshevCFT
import EnvelopeApproximation.ChebyshevCFT: fourier_mode, scale, translation
using StaticArrays
using LinearAlgebra
using Intervals
import Intervals: IntervalSet
using Rotations
import Base: *, ∈, isempty, ~, ∩, convert, ⊆
using QuadGK
using IterTools

export IntersectionDome, intersection_domes, *, ⊆
export unfold_periodic_bubbles, append_periodic_bubbles!, original_bubble_groups
export PeriodicInterval, mod2π, ∈, EmptyArc, FullCircle, complement, a, b
export ring_domes_complement_intersection!

include("Rotations.jl")
include("IntersectionDome.jl")
include("PeriodicCopies.jl")
include("PeriodicInterval.jl")
include("RingDomesComplementIntersection.jl")
include("PolarLimits.jl")

end