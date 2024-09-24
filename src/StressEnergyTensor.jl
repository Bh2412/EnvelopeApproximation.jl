module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubbleBasics: ⋅
using EnvelopeApproximation.BubblesEvolution
using Base.Iterators
import EnvelopeApproximation.SurfaceTesselation: surface_sections, BubbleSection, unit_sphere_point, bubble_point
using EnvelopeApproximation.SurfaceTesselation
using StaticArrays
using HCubature
import HCubature: hcubature
import Base: exp, push!

TensorDirection = Union{Symbol, Tuple{Symbol, Symbol}}

export TensorDirection

unit_sphere_point(x:: SVector{2, Float64}):: Point3 = unit_sphere_point(x...)

function td_integrand(x:: SVector{2, Float64}, s:: Symbol):: Float64
    if s ≡ :trace
        return 1.
    elseif s ≡ :x
        return cos(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :y
        return sin(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :z
        return x[2]
    end
end

function td_integrand(x:: SVector{2, Float64}, td:: Tuple{Symbol, Symbol}):: Float64 
    td ≡ (:x, :x) && return cos(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:y, :y) && return sin(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:z, :z) && return x[2] ^ 2
    ((td ≡ (:x, :y)) | (td ≡ (:y, :x))) && return cos(x[1]) * sin(x[1]) * (1 - x[2] ^ 2)
    return td_integrand(x, td[1]) * td_integrand(x, td[2])
end

⋅(p1:: Point3, k:: Vec3):: Float64 = coordinates(p1) ⋅ k
⋅(x:: SVector{2, Float64}, k:: Vec3) = (√(1 - x[2] ^ 2) * (k[1]  * cos(x[1]) + k[2] * sin(x[1])) + k[3] * x[2])

struct BubbleIntegrand!
    bubble_intersection:: BubbleIntersection
    bubble:: Bubble
    ΔV:: Float64
end

function (bi:: BubbleIntegrand!)(x:: SVector{2, Float64}, 
                                 ks:: Vector{Vec3},
                                 tensor_directions:: Vector{TensorDirection},
                                 V:: Matrix{ComplexF64}, 
                                 e_dump:: Vector{ComplexF64},
                                 td_dump:: Vector{Float64})
    ∉(x[2], x[1], bi.bubble_intersection) && return
    p = bubble_point(x..., bi.bubble)
    c = bi.bubble.radius ^ 3 * ((bi.ΔV / 3.))
    @. e_dump = cis(-((p, ) ⋅ ks))
    @. td_dump = td_integrand((x, ), tensor_directions) * c
    @. V += $reshape(e_dump, :, 1) * $reshape(td_dump, 1, :)
end

struct SurfaceIntegrand
    bubble_integrands:: Vector{BubbleIntegrand!}
    ks:: Vector{Vec3}
    tensor_directions:: Vector{TensorDirection}
    V:: Matrix{ComplexF64}
    e_dump:: Vector{ComplexF64}
    td_dump:: Vector{Float64}

    function SurfaceIntegrand(bubble_integrands:: Vector{BubbleIntegrand!},
                               ks:: Vector{Vec3},
                               tensor_directions:: Vector{TensorDirection})
        V = zeros(ComplexF64, length(ks), length(tensor_directions))
        e_dump = zeros(ComplexF64, length(ks))
        td_dump = zeros(Float64, length(tensor_directions))
        return new(bubble_integrands, ks, tensor_directions,
                   V, e_dump, td_dump)
    end
end

function (si:: SurfaceIntegrand)(x:: SVector{2, Float64}):: Matrix{ComplexF64}
    V = zeros(ComplexF64, length(si.ks), length(si.tensor_directions))
    for bi! in si.bubble_integrands
        bi!(x, si.ks, si.tensor_directions, V, si.e_dump, si.td_dump)
    end
    return V
end

push!(si:: SurfaceIntegrand, bi:: BubbleIntegrand!) = push!(si.bubble_integrands, bi)

function unit_sphere_hcubature(f; kwargs...)
    x0, xf = SVector{2, Float64}(0., -1.), SVector{2, Float64}(2π, 1.)
    return hcubature(f, x0, xf; kwargs...)
end

function surface_integral(ks:: Vector{Vec3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector{TensorDirection},
                          intersections:: Dict{Int64, BubbleIntersection},
                          ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    surface_integrand = SurfaceIntegrand(Vector{BubbleIntegrand!}(), 
                                         ks, 
                                         tensor_directions)
    for (bubble_index, bubble_intersection) in intersections
        push!(surface_integrand, BubbleIntegrand!(bubble_intersection, 
                                                  bubbles[bubble_index],
                                                  ΔV))
    end
    return unit_sphere_hcubature(surface_integrand; kwargs...)[1]
end

function surface_integral(ks:: Vector{Vec3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector{TensorDirection},
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    intersections = bubble_intersections(ϕ_resolution, μ_resolution, bubbles)
    surface_integral(ks, bubbles, tensor_directions, intersections, ΔV; kwargs...)
end

function surface_integral(ks:: Vector{Vec3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          n_ϕ:: Int64, 
                          n_μ:: Int64,
                          ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    return surface_integral(ks, bubbles, tensor_directions, 
                            2π / n_ϕ, 2. / n_μ, ΔV; kwargs...)
end

export surface_integral

struct BubblePotentialIntegrand!
    bubble_intersection:: BubbleIntersection
    bubble:: Bubble
    ΔV:: Float64
end

function (bpi:: BubblePotentialIntegrand!)(x:: SVector{2, Float64}, 
                                           ks:: Vector{Vec3},
                                           V:: Vector{ComplexF64})
    ∉(x[2], x[1], bpi.bubble_intersection) && return
    p = bubble_point(x..., bpi.bubble)
    c = (bpi.bubble.radius ^ 2)
    @. V += cis(-((p, ) ⋅ ks)) * ((x, ) ⋅ ks) * c
end

struct PotentialIntegrand
    bubble_potential_integrands:: Vector{BubblePotentialIntegrand!}
    ks:: Vector{Vec3}
    V:: Vector{ComplexF64}

    function PotentialIntegrand(bubble_potential_integrands:: Vector{BubblePotentialIntegrand!},
                                ks:: Vector{Vec3})
        V = zeros(ComplexF64, length(ks))
        return new(bubble_potential_integrands, ks, V)
    end
end

function (bpi:: PotentialIntegrand)(x:: SVector{2, Float64}):: Vector{ComplexF64}
    V = zeros(ComplexF64, length(bpi.ks))
    for bpi! in bpi.bubble_potential_integrands
        bpi!(x, bpi.ks, V)
    end
    return V
end

push!(pi:: PotentialIntegrand, bpi:: BubblePotentialIntegrand!) = push!(pi.bubble_potential_integrands, bpi)

function potential_integral(ks:: Vector{Vec3}, 
                            bubbles:: Bubbles, 
                            intersections:: Dict{Int, BubbleIntersection},
                            ΔV:: Float64 = 1.; kwargs...):: Vector{ComplexF64}
    potential_integrand = PotentialIntegrand(Vector{BubblePotentialIntegrand!}(), 
                                             ks)
    for (bubble_index, bubble_intersection) in intersections
        push!(potential_integrand, BubblePotentialIntegrand!(bubble_intersection, 
                                                             bubbles[bubble_index],
                                                             ΔV))
    end
    c:: Vector{ComplexF64} = @. im * ((-ΔV) / (ks ⋅ ks))
    return @. c * $unit_sphere_hcubature(potential_integrand; kwargs...)[1]
end

function potential_integral(ks:: Vector{Vec3}, 
                            bubbles:: Bubbles, 
                            ϕ_resolution:: Float64,
                            μ_resolution:: Float64,
                            ΔV:: Float64 = 1.; kwargs...):: Vector{ComplexF64}
    intersections = bubble_intersections(ϕ_resolution, μ_resolution, bubbles)
    return potential_integral(ks, bubbles, intersections, ΔV; kwargs...)
end

function potential_integral(ks:: Vector{Vec3}, 
                            bubbles:: Bubbles, 
                            n_ϕ:: Int64, 
                            n_μ:: Int64, 
                            ΔV:: Float64 = 1.; kwargs...):: Vector{ComplexF64}
    return potential_integral(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV; kwargs...)
end

export potential_integral

CARTESIAN_DIRECTIONS = [:x, :y, :z]
diagonal = [(s, s) for s in CARTESIAN_DIRECTIONS]
above_diagonal = [(:x, :y), (:x, :z), (:y, :z)]
upper_right = vcat(diagonal, above_diagonal)

function T_ij(ks:: Vector{Vec3}, 
              bubbles:: Bubbles, 
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector{TensorDirection}, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 2}
    isnothing(tensor_directions) && (tensor_directions = Vector{TensorDirection}(vcat([:trace], upper_right)))
    intersections = bubble_intersections(ϕ_resolution, μ_resolution, bubbles)
    T = surface_integral(ks, bubbles, tensor_directions, intersections, 
                         ΔV; kwargs...)
     # Based on the following:
    ```math
    T_ij = ∂_iϕ∂_jϕ - δ_ij⋅L ≈ ∂_iϕ∂_jϕ - δ_ij V
    ```         
    if any(td in tensor_directions for td in diagonal)
        vi = potential_integral(ks, bubbles, intersections, ΔV; kwargs...)
        for (i, td) in enumerate(tensor_directions)
            if td in diagonal
                T[:, i] .-= vi
            end
        end
    end
    return T
end

function T_ij(ks:: Vector{Vec3}, 
              bubbles:: Bubbles, 
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 2}
    return T_ij(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions; kwargs...)
end

function T_ij(ks:: Vector{Vec3}, snapshot:: BubblesSnapShot, times:: Vector{Float64}, 
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector{TensorDirection}, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 3}
    _bubbles(t) = current_bubbles(at_earlier_time(snapshot, t))
    M = Array{ComplexF64, 3}(undef, length(times), length(ks), length(tensor_directions))
    for (i, t) in enumerate(times)
        bs = _bubbles(t)
        M[i, :, :] .= T_ij(ks, bs, ϕ_resolution, μ_resolution, ΔV, tensor_directions; kwargs...)
    end 
    return M
end

function T_ij(ks:: Vector{Vec3}, snapshot:: BubblesSnapShot, times:: Vector{Float64}, 
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector{TensorDirection}, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 3}
    return T_ij(ks, snapshot, times, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions; kwargs...)
end

export T_ij

end