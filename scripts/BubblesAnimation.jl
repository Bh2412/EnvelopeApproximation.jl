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
_snap = ensemble[10, 1]
snap = Observable(_snap)
function _viz(x:: Observable{BubblesSnapShot}; kw...)
    if length(x[].nucleations) == 0.
        return viz(Meshes.Ball(Meshes.Point3(0.,0.,0.), 0.); kw...)
    end
    return viz(x[]; kw...)
end

ball(bubble:: Bubble):: Meshes.Ball = Meshes.Ball(bubble.center, bubble.radius)
bubbles_mesh(bubbles:: Bubbles) = Meshes.union(ball.(bubbles))
bubbles_mesh(snap:: BubblesSnapShot) = bubbles_mesh(current_bubbles(snap))

function gset(snap:: BubblesSnapShot):: Meshes.GeometrySet{3, Float64}
    if length(snap.nucleations) == 0
        return Meshes.GeometrySet{3, Float64}([Meshes.Ball(Meshes.Point3(0., 0., 0.), 0.)])
    end
    return Meshes.GeometrySet(bubbles_mesh(snap))
end
@show at_earlier_time(snap[], 0.)
a = _viz(snap)
_gset = a.plot[1]
figure = a.figure
time_slider = Slider(figure[2, 1], range=0.:0.1:_snap.t)
@show gset(at_earlier_time(_snap, 0.))
on(time_slider.value) do t
     _gset[] = gset(at_earlier_time(_snap, t))
end
