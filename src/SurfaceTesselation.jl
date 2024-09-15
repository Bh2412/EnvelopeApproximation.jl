module SurfaceTesselation
using EnvelopeApproximation.BubbleBasics
import Base
import Base: *, +, ∈, ∉, push!
import Distances.pairwise

n(range:: Float64, resolution:: Float64):: Int64 = ceil(Int64, range / resolution)
n(t:: Tuple{Float64, Float64}):: Int64 = n(t[1], t[2])
ns(ϕ_resolution:: Float64, μ_resolution:: Float64):: Vector{Int} = n.([(2π, ϕ_resolution), (2., μ_resolution)])

function bounds(ϕ_resolution:: Float64, μ_resolution:: Float64):: Tuple{LinRange{Float64, Int64}, LinRange{Float64, Int64}}
    n_ϕ, n_μ = ns(ϕ_resolution, μ_resolution)
    return LinRange(0., 2π, n_ϕ + 1), LinRange(-1., 1., n_μ + 1)
end

middle(bounds:: LinRange{Float64}):: LinRange{Float64} = (bounds[2:end] + bounds[1:(end - 1)]) / 2

struct Section
    c:: Float64
    d:: Float64
end

function unit_sphere_tesselation(ϕ_resolution:: Float64, μ_resolution:: Float64):: Tuple{Vector{Section}, Vector{Section}}
    n_ϕ, n_μ = ns(ϕ_resolution, μ_resolution)
    ϕ, μ = middle.(bounds(ϕ_resolution, μ_resolution))
    return Section.(ϕ, (2π / n_ϕ, )), Section.(μ, (2. / n_μ, ))
end

*(point:: Point3, r:: Float64):: Point3 = Point3(r .* coordinates(point))
+(point1:: Point3, point2:: Point3):: Point3 = Point3(coordinates(point1) + coordinates(point2))

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

function surface_sections(us_sections:: Vector{UnitSphereSection}, bubbles:: Bubbles):: Vector{BubbleSection}
    pss = preliminary_surface_sections(us_sections, bubbles)
    filt = sum(reshape(bubbles.bubbles, :, 1) .∋ reshape(bubble_point.(pss, (bubbles, )), 1, :), dims=1) .<= 1
    return [p for (i, p) in enumerate(pss) if filt[i]]
end

function surface_sections(ϕ_resolution:: Float64, μ_resolution:: Float64, bubbles:: Bubbles)::  Vector{BubbleSection}
    return surface_sections(unit_sphere_sections(ϕ_resolution, μ_resolution), bubbles)
end

surface_sections(n_ϕ:: Int64, n_μ:: Int64, bubbles:: Bubbles):: Vector{BubbleSection} = surface_sections(2π / n_ϕ, 2. / n_μ, bubbles)

export surface_sections

struct BubbleIntersection
    μ:: Vector{Float64}
    ϕ:: Vector{Float64}
    bubble_index:: Int64
    included:: Set{CartesianIndex{2}}
    
    function BubbleIntersection(μ:: AbstractVector{Float64}, ϕ:: AbstractVector{Float64}, bubble_index:: Int, included:: Union{Set{CartesianIndex{2}}, Nothing} = nothing)
        included ≡ nothing && (included = Set{CartesianIndex{2}}())
        return new(sort(μ), sort(ϕ), bubble_index, included)
    end
end

export BubbleIntersection

push!(bi:: BubbleIntersection, ind:: CartesianIndex{2}) = push!(bi.included, ind)

idx(μ:: Float64, ϕ:: Float64, μ_limits:: AbstractVector{Float64}, ϕ_limits:: AbstractVector{Float64}):: CartesianIndex{2} = begin
    μ_index = searchsortedfirst(μ_limits, μ, lt=<=) - 1
    ϕ_index = searchsortedfirst(ϕ_limits, ϕ, lt=<=) - 1
    μ ≈ μ_limits[end] && (μ_index = length(μ_limits) - 1)
    ϕ ≈ ϕ_limits[end] && (ϕ_index = length(ϕ_limits) - 1)
    CartesianIndex(μ_index, ϕ_index)
end

idx(s:: BubbleSection, μ_limits:: AbstractVector{Float64}, ϕ_limits:: AbstractVector{Float64}):: CartesianIndex{2} = idx(s.μ.c, s.ϕ.c, μ_limits, ϕ_limits)
idx(μ:: Float64, ϕ:: Float64, bi:: BubbleIntersection):: CartesianIndex{2} = idx(μ, ϕ, bi.μ, bi.ϕ)
idx(s:: BubbleSection, bi:: BubbleIntersection):: CartesianIndex{2} = idx(s, bi.μ, bi.ϕ)

function ∈(μ:: Float64, ϕ:: Float64, bi:: BubbleIntersection):: Bool
    return idx(μ, ϕ, bi) ∈ bi.included
end

function ∉(μ:: Float64, ϕ:: Float64, bi:: BubbleIntersection):: Bool
    return idx(μ, ϕ, bi) ∉ bi.included
end

export ∈, ∉

function bubble_intersections(ϕ_resolution:: Float64, μ_resolution:: Float64, bubbles:: Bubbles):: Dict{Int64, BubbleIntersection}
    sections = surface_sections(ϕ_resolution, μ_resolution, bubbles)
    ϕ_limits, μ_limits = bounds(ϕ_resolution, μ_resolution)
    d = Dict{Int64, BubbleIntersection}()
    for section ∈ sections
        if section.bubble_index ∉ keys(d)
            d[section.bubble_index] = BubbleIntersection(μ_limits, ϕ_limits, section.bubble_index)
        end
        push!(d[section.bubble_index], idx(section, μ_limits, ϕ_limits))
    end
    return d
end

function bubble_intersections(n_ϕ:: Int64, n_μ:: Int64, bubbles:: Bubbles):: Dict{Int64, BubbleIntersection}
    return bubble_intersections(2π / n_ϕ, 2. / n_μ, bubbles)
end

export bubble_intersections

end