module VolumeIntegration

using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.n
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.middle
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.*
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.+
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.unit_sphere_point
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.euc
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration.BubblePoint
using Distances

ns(v_resoluiton:: Float64, ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{Int} = n.([(1. /3, v_resoluiton), (2π, ϕ_resolution), (2., μ_resolution)])

function unit_ball_tesselation(v_resoluiton:: Float64, 
                               ϕ_resolution:: Float64, 
                               μ_resolution:: Float64):: Vector{LinRange{Float64}}
    _ns = ns(v_resoluiton, ϕ_resolution, μ_resolution)
    ranges = [(0., 1. /3), (0., 2π), (-1., 1.)]
    return [middle(LinRange(range..., _n + 1)) for (range, _n) in zip(ranges, _ns)]
end

function unit_ball_point(v:: Float64, ϕ:: Float64, μ:: Float64):: Point3
    usp = unit_sphere_point(ϕ, μ)
    r = (3 * v) ^ (1. / 3)
    return usp * r
end

function unit_ball_points(v_resoluiton:: Float64, 
                          ϕ_resolution:: Float64, 
                          μ_resolution:: Float64)
    vs, ϕs, μs = unit_ball_tesselation(v_resoluiton, ϕ_resolution, μ_resolution)
    return [unit_ball_point(v, ϕ, μ) for v in vs for ϕ in ϕs for μ in μs]
end

function volume_points(bubbles:: Bubbles, 
                       v_resoluiton:: Float64, 
                       ϕ_resolution:: Float64, 
                       μ_resolution:: Float64)
    ub_points = unit_ball_points(v_resoluiton, ϕ_resolution, μ_resolution)
    ps = (reshape(ub_points, (length(ub_points), 1)) .* reshape(radii(bubbles), (1, length(bubbles)))) .+ reshape(centers(bubbles), (1, length(bubbles)))
    return reshape([BubblePoint(p, i[2]) for (i, p) in pairs(ps)], length(bubbles) * length(ub_points))
end

function weights(ps:: Vector{BubblePoint}, bubbles:: Bubbles):: Vector{Float64}
    dm = pairwise(euc, centers(bubbles), ps)
    return 1 ./ reshape(sum(dm .≤ reshape(radii(bubbles), (length(bubbles), 1)), dims=1), length(ps))
end

function volume_integral(f:: Function, bubbles:: Bubbles, 
                         v_resolution:: Float64, 
                         ϕ_resolution:: Float64, 
                         μ_resolution:: Float64)
    ps = volume_points(bubbles, v_resolution, ϕ_resolution, μ_resolution)
    _weights = weights(ps, bubbles)
    section_volumes = begin
        volumes = (4π / 3.) .* (radii(bubbles) .^ 3)
        N = prod(ns(v_resolution, ϕ_resolution, μ_resolution))
        volumes ./ N
    end
    return sum(f(p.point) * weight * section_volumes[p.bubble_index]
               for (p, weight) in zip(ps, _weights))
end

end