using JLD2
using GLMakie
using EnvelopeApproximation
using EnvelopeApproximation.Visualization
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
import Meshes
using Observables

@show pwd()
ensemble = load("evolution_ensemble.jld2", "snapshots")
snap = ensemble[10, 1]
viz(snap)
