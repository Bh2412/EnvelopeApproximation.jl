module Visualization

using Meshes
using GLMakie
import Meshes.viz
import Meshes.viz!
import Meshes.Point
using EnvelopeApproximation.BubbleBasics: Point3 as P3, Bubble, Bubbles
using EnvelopeApproximation.BubblesEvolution

Point(p:: P3) = Point(p.coordinates...)

ball(bubble:: Bubble):: Ball = Ball(Point(bubble.center), bubble.radius)
bubbles_mesh(bubbles:: Bubbles) = union(ball.(bubbles))
viz(bubble:: Bubble; kw...) = viz(ball(bubble); kw...)
viz!(bubble:: Bubble; kw...) = viz!(ball(bubble); kw...) 
viz(bubbles:: Bubbles; kw...) = viz(bubbles_mesh(bubbles); kw...)
viz!(bubbles:: Bubbles; kw...) = viz!(bubbles_mesh(bubbles); kw...) 
viz(snap:: BubblesSnapShot; kw...) = viz(current_bubbles(snap), kw...)
viz!(snap:: BubblesSnapShot; kw...) = viz!(current_bubbles(snap), kw...)
export viz
export viz!

end