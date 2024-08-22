module SurfaceIntegration
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesIntegration
import Base
import Base: *, +
import EnvelopeApproximation.BubblesEvolution.euc
import Meshes.coordinates
import Distances.pairwise

n(range:: Float64, resolution:: Float64):: Int64 = ceil(Int64, range / resolution)
n(t:: Tuple{Float64, Float64}):: Int64 = n(t[1], t[2])
ns(ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{Int} = n.([(2π, ϕ_resolution), (2., μ_resolution)])

middle(bounds:: LinRange{Float64}):: LinRange{Float64} = (bounds[2:end] + bounds[1:(end - 1)]) / 2

struct Section
    c:: Float64
    d:: Float64
end

function unit_sphere_tesselation(ϕ_resolution:: Float64, μ_resolution:: Float64):: Tuple{Vector{Section}, Vector{Section}}
    n_ϕ, n_μ = ns(ϕ_resolution, μ_resolution)
    ϕ = middle(LinRange(0., 2π, n_ϕ + 1))
    μ = middle(LinRange(-1., 1., n_μ + 1))
    return Section.(ϕ, (2π / n_ϕ, )), Section.(μ, (2. / n_μ, ))
end

*(point:: Point3, r:: Float64):: Point3 = Point3(r .* point.coords)
+(point1:: Point3, point2:: Point3):: Point3 = Point3(point1.coords + point2.coords)

function unit_sphere_point(ϕ:: Float64, μ:: Float64):: Point3
    s = sqrt((1. - μ ^ 2))
    return Point3(s * cos(ϕ), s * sin(ϕ), μ)
end

unit_sphere_point(ϕ:: Section, μ:: Section) = unit_sphere_point(ϕ.c, μ.c)

export unit_sphere_point

struct UnitSphereSection
    ϕ:: Section 
    μ:: Section
end

function bubble_point(ϕ:: Float64, μ:: Float64, bubble:: Bubble):: Point3
    usp = unit_sphere_point(ϕ, μ)
    return usp * bubble.radius + bubble.center
end

bubble_point(ϕ:: Section, μ:: Section, bubble:: Bubble) = bubble_point(ϕ.c, μ.c, bubble)
bubble_point(ϕ:: Float64, μ:: Float64, bubble_index:: Int, bubbles:: Bubbles) = bubble_point(ϕ, μ, bubbles[bubble_index])
bubble_point(ϕ:: Section, μ:: Section, bubble_index:: Int, bubbles:: Bubbles) = bubble_point(ϕ.c, μ.c, bubbles[bubble_index])

struct BubbleSection
    ϕ:: Section 
    μ:: Section
    bubble_index:: Int
end

function BubbleSection(sphere_s:: UnitSphereSection, bubble_index:: Int)
    return BubbleSection(sphere_s.ϕ, sphere_s.μ, bubble_index)
end

bubble_point(bubble_section:: BubbleSection, bubbles:: Bubbles) = bubble_point(bubble_section.ϕ, bubble_section.μ, 
                                                                               bubble_section.bubble_index, bubbles)

function unit_sphere_sections(ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{UnitSphereSection}
    ϕ, μ = unit_sphere_tesselation(ϕ_resolution, μ_resolution)
    return [UnitSphereSection(_ϕ, _μ) for _ϕ in ϕ for _μ in μ]
end


function preliminary_surface_sections(us_sections:: Vector{UnitSphereSection}, bubbles:: Bubbles)
    return [BubbleSection(sphere_s, i) for sphere_s in us_sections for i in eachindex(bubbles)]
end

function ≲(a:: Float64, b:: Float64):: Bool
    return (a <= b) | (a ≈ b)
end

function surface_sections(us_sections:: Vector{UnitSphereSection}, bubbles:: Bubbles):: Vector{BubbleSection}
    pss = preliminary_surface_sections(us_sections, bubbles)
    dm = pairwise(euc, centers(bubbles), bubble_point.(pss, (bubbles, )))
    filt = sum((dm .≲ reshape(radii(bubbles), (length(bubbles), 1))), dims=1) .<= 1
    return [p for (i, p) in enumerate(pss) if filt[i]]
end

function surface_sections(ϕ_resolution:: Float64, μ_resolution:: Float64, bubbles:: Bubbles)::  Vector{BubbleSection}
    return surface_sections(unit_sphere_sections(ϕ_resolution, μ_resolution), bubbles)
end

function surface_integral(f:: Function, bubbles:: Bubbles, ϕ_resolution:: Float64, μ_resolution:: Float64)
    ps = surface_sections(ϕ_resolution, μ_resolution, bubbles)
    section_areas = begin 
        surface_areas = 4π * (radii(bubbles) .^ 2)
        N = prod(ns(ϕ_resolution, μ_resolution))
        surface_areas ./ N
    end
    return sum(f(p) .* section_areas[p.bubble_index] for p in ps)
end

surface_integral(f:: Function, bubbles:: Bubbles, n_ϕ:: Int64, n_μ:: Int64) = surface_integral(f, bubbles, 2π / n_ϕ, 2. / n_μ)

end
