using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.StressEnergyTensor
import LinearAlgebra: norm
import Meshes: coordinates
using GLMakie

Point3 = EnvelopeApproximation.Point3

# Setting up the bubbles
R = 1.
bubbles = Bubbles([Bubble(Point3(0., 0., 0.), R)])

# Setting up the ks

k_0 = 2π / R
ks = LinRange(k_0 / 10, k_0 * 10, 1000)
k_vecs = (x -> Point3(0., 0., x)).(ks)
norm(p:: Point3) = norm(coordinates(p), 2)

# Computing Analytically

ΔV = 1.
analytic_T_ii = @. ((ΔV * 4π / 3) * (R ^ 3)) * sin(ks * R) / (ks * R)  
lines(ks, analytic_T_ii)

# Computing Numerically
#tensor_directions = [:trace, (:x, :x), (:y, :y), (:z, :z), (:x, :y), (:x, :z), (:y, :z)]
tensor_directions = [:trace]
surface_integral = EnvelopeApproximation.StressEnergyTensor.surface_integral
numerical_T_ii = surface_integral(k_vecs, bubbles, tensor_directions, 100, 100, ΔV)
lines(ks, numerical_T_ii[:, 1] .|> real)
lines!(ks, analytic_T_ii)