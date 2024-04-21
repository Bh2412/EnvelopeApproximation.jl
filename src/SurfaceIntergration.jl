module SurfaceIntegration
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesIntegration
import Base.*
import Base.+
import EnvelopeApproximation.BubblesEvolution.euc
import Meshes.coordinates
import Distances.pairwise

n(range:: Float64, resolution:: Float64):: Int64 = ceil(Int64, range / resolution)
n(t:: Tuple{Float64, Float64}):: Int64 = n(t[1], t[2])
ns(ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{Int} = n.([(2π, ϕ_resolution), (2., μ_resolution)])

middle(bounds:: LinRange{Float64}):: LinRange{Float64} = (bounds[2:end] + bounds[1:(end - 1)]) / 2

function unit_sphere_tesselation(ϕ_resolution:: Float64, μ_resolution:: Float64):: Tuple{LinRange{Float64}, LinRange{Float64}}
    n_ϕ, n_μ = ns(ϕ_resolution, μ_resolution)
    ϕ = middle(LinRange(0., 2π, n_ϕ + 1))
    μ = middle(LinRange(-1., 1., n_μ + 1))
    return ϕ, μ
end

function unit_sphere_point(ϕ:: Float64, μ:: Float64):: Point3
    s = sqrt((1. - μ ^2))
    return Point3(s * cos(ϕ), s * sin(ϕ), μ)
end

function unit_sphere_points(ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{Point3}
    ϕ, μ = unit_sphere_tesselation(ϕ_resolution, μ_resolution)
    return [unit_sphere_point(_ϕ, _μ) for _ϕ in ϕ for _μ in μ]
end

*(point:: Point3, r:: Float64):: Point3 = Point3(r .* point.coords)
+(point1:: Point3, point2:: Point3):: Point3 = Point3(point1.coords + point2.coords)


function _preliminary_surface_points(us_points:: Vector{Point3}, bubbles:: Bubbles):: Array{Point3, 2}
    return (reshape(us_points, (length(us_points), 1)) .* reshape(radii(bubbles), (1, length(bubbles)))) .+ reshape(centers(bubbles), (1, length(bubbles)))
end

struct BubblePoint
    point:: Point3
    bubble_index:: Int
end

coordinates(p:: BubblePoint) = coordinates(p.point)

export coordinates

function preliminary_surface_points(us_points:: Vector{Point3}, bubbles:: Bubbles):: Vector{BubblePoint}
    _psp = _preliminary_surface_points(us_points, bubbles)
    return reshape([BubblePoint(p, i[2]) for (i, p) in pairs(_psp)], length(bubbles) * length(us_points))
end

function preliminary_surface_points(ϕ_resolution:: Float64, μ_resolution:: Float64, bubbles:: Bubbles):: Vector{BubblePoint}
    return preliminary_surface_points(unit_sphere_points(ϕ_resolution, μ_resolution), bubbles)
end

euc(point:: Point3, bubble_point:: BubblePoint):: Float64 = euc(point, bubble_point.point)

function surface_points(us_points:: Vector{Point3}, bubbles:: Bubbles):: Vector{BubblePoint}
    psp =  preliminary_surface_points(us_points, bubbles)
    dm = pairwise(euc, centers(bubbles), psp)
    filt = sum((dm .≤ reshape(radii(bubbles), (length(bubbles), 1))), dims=1) .<= 1
    return [p for (i, p) in enumerate(psp) if filt[i]]
end

function surface_points(ϕ_resolution:: Float64, μ_resolution:: Float64, bubbles:: Bubbles)::  Vector{BubblePoint}
    return surface_points(unit_sphere_points(ϕ_resolution, μ_resolution), bubbles)
end

function surface_integral(f:: Function, bubbles:: Bubbles, ϕ_resolution:: Float64, μ_resolution:: Float64)
    ps = surface_points(ϕ_resolution, μ_resolution, bubbles)
    section_areas = begin 
        surface_areas = 4π * (radii(bubbles) .^ 2)
        N = prod(ns(ϕ_resolution, μ_resolution))
        surface_areas ./ N
    end
    return sum(f(p) .* section_areas[p.bubble_index] for p in ps)
end

surface_integral(f:: Function, bubbles:: Bubbles, n_ϕ:: Int64, n_μ:: Int64) = surface_integral(f, bubbles, 2π / n_ϕ, 2. / n_μ)

end
