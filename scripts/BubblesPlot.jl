using JLD2
using GLMakie
using EnvelopeApproximation
using EnvelopeApproximation.Visualization
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
import Meshes
using Observables

@show pwd()
ensemble = load("./notebooks/evolution_ensemble.jld2", "space_size")
snap = ensemble[10, 1]
viz(snap)
