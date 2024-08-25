module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using Base.Iterators
import Base./
import Meshes: Vec, Point3, coordinates, ⋅, -
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration: surface_integral, BubbleSection, unit_sphere_point, bubble_point
using LinearAlgebra
using StaticArrays
using HCubature
import HCubature: hcubature

TensorDirection{N} = Union{Symbol, NTuple{N, Symbol}} where N

export TensorDirection

unit_sphere_point(x:: SVector{2, Float64}) = unit_sphere_point(x...)

function td_integrand(x:: SVector{2, Float64}, tensor_direction):: Float64 
    if tensor_direction ≡ :trace
        return 1.
    end
    CARTESIAN_DIRECTIONS = [:x, :y, :z]
    try
        indices:: Vector{Int64} = indexin(tensor_direction, CARTESIAN_DIRECTIONS)
        return prod(coordinates(unit_sphere_point(x))[indices])
    catch e
        if e isa MethodError
            throw(e("All directions must be elements of $CARTESIAN_DIRECTIONS"))
        else
            throw(e)
        end
    end
end

function td_integrand(x:: SVector{2, Float64}, tensor_directions:: Vector):: Vector{Float64}
    return td_integrand.((x, ), tensor_directions)
end

⋅(p1:: Point3, p2:: Point3):: Float64 = ⋅(coordinates.([p1, p2])...)
/(p:: Point3, d:: Float64):: Point3 = Point3((coordinates(p) / d)...)
_exp(p:: Point3, k:: Point3) = exp(-im * (p ⋅ k))

function surface_integrand(ks:: Vector{Point3}, bubbles:: Bubbles, tensor_directions:: Vector, 
                           ΔV:: Float64 = 1.)
    ks = reshape(ks, (length(ks), 1))
    _td_integrand = (a -> reshape(a, (1, length(tensor_directions)))) ∘ (x -> td_integrand(x, tensor_directions)) 
    c = (ΔV / 3)
    function _integrand(x:: SVector{2, Float64}, bubble_index:: Int):: Array{Complex, 2}
        p = bubble_point(x..., bubble_index, bubbles)
        return @. (_exp((p, ), ks)) * ($_td_integrand(x) * (bubbles[bubble_index].radius * c))
    end>
    return _integrand
end

measure(p:: BubbleSection) = p.ϕ.d * p.μ.d
hcubature(f, p:: BubbleSection; kwargs...) = hcubature(f, [p.ϕ.c - p.ϕ.d / 2, p.μ.c - p.μ.d / 2], [p.ϕ.c + p.ϕ.d / 2, p.μ.c + p.μ.d / 2]; kwargs...)

function surface_section_average(ks:: Vector{Point3}, bubbles:: Bubbles, tensor_directions:: Vector, 
                                 ΔV:: Float64 = 1.; kwargs...)
    integrand = surface_integrand(ks, bubbles, tensor_directions, ΔV)
    function _section_average(p:: BubbleSection):: Array{Complex, 2}
        return (1 / measure(p)) * hcubature(x -> integrand(x, p.bubble_index), p; kwargs...)[1]
    end 
    return _section_average
end

function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          ΔV:: Float64 = 1.; kwargs...):: Array{ComplexF64, 2}
    integrand = surface_section_average(ks, bubbles, tensor_directions, ΔV; kwargs...)
    return surface_integral(integrand, bubbles, ϕ_resolution, μ_resolution)
end

function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          n_ϕ:: Int64, 
                          n_μ:: Int64,
                          ΔV:: Float64 = 1.; kwargs...)
    return surface_integral(ks, bubbles, tensor_directions, 
                            2π / n_ϕ, 2. / n_μ, ΔV; kwargs...)
end

export surface_integral

unit_sphere_point(p:: BubbleSection) = unit_sphere_point(p.ϕ, p.μ)

function element_projection(x:: SVector{2, Float64}, ks:: Vector{Point3}):: Vector{Float64}
    return unit_sphere_point(x) .⋅ ks
end

function potential_integrand(ks:: Vector{Point3}, bubbles:: Bubbles,
                             ΔV:: Float64 = 1.):: Function
    projection = x -> element_projection(x, ks)
    #=
    This assumes that the potential is negative within the true vacuum
    and zero outside of it.
    =#
    c = @. im * ((-ΔV) / (ks ⋅ ks))
    function integrand(x:: SVector{2, Float64}, bubble_index:: Int):: Vector{ComplexF64}
        p = bubble_point(x..., bubble_index, bubbles)
        return @. _exp((p, ), ks) * $projection(x) * c
    end
end

function potential_section_average(ks:: Vector{Point3}, bubbles:: Bubbles,
                                   ΔV:: Float64 = 1.; kwargs...)
    integrand = potential_integrand(ks, bubbles, ΔV)
    function _section_average(p:: BubbleSection):: Vector{ComplexF64}
        return (1 / measure(p)) * hcubature(x -> integrand(x, p.bubble_index), p; kwargs...)[1]
    end 
    return _section_average
end

function potential_integral(ks:: Vector{Point3}, 
                            bubbles:: Bubbles, 
                            ϕ_resolution:: Float64, 
                            μ_resolution:: Float64, 
                            ΔV:: Float64 = 1.; kwargs...)
    integrand = potential_section_average(ks, bubbles, ΔV; kwargs...)
    return surface_integral(integrand, bubbles, ϕ_resolution, μ_resolution)
end

function potential_integral(ks:: Vector{Point3}, 
                            bubbles:: Bubbles, 
                            n_ϕ:: Int64, 
                            n_μ:: Int64, 
                            ΔV:: Float64 = 1.; kwargs...)
    return potential_integral(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV; kwargs...)
end

export potential_integral

CARTESIAN_DIRECTIONS = [:x, :y, :z]
diagonal = [(s, s) for s in CARTESIAN_DIRECTIONS]
above_diagonal = [(:x, :y), (:x, :z), (:y, :z)]
upper_right = vcat(diagonal, above_diagonal)

function T_ij(ks:: Vector{Point3}, 
              bubbles:: Bubbles, 
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 2}
    isnothing(tensor_directions) && (tensor_directions = vcat([:trace], upper_right))
    T = surface_integral(ks, bubbles, tensor_directions, ϕ_resolution,
                         μ_resolution, ΔV; kwargs...)
     # Based on the following:
    ```math
    T_ij = ∂_iϕ∂_jϕ - δ_ij⋅L ≈ ∂_iϕ∂_jϕ - δ_ij V
    ```         
    if any(td in tensor_directions for td in diagonal)
        vi = potential_integral(ks, bubbles, ϕ_resolution, 
                                μ_resolution, ΔV; kwargs...)
        for (i, td) in enumerate(tensor_directions)
            if td in diagonal
                T[:, i] .-= vi
            end
        end
    end
    return T
end

function T_ij(ks:: Vector{Point3}, 
              bubbles:: Bubbles, 
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 2}
    return T_ij(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions; kwargs...)
end

function T_ij(ks:: Vector{Point3}, snapshot:: BubblesSnapShot, times:: Vector{Float64}, 
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 3}
    _bubbles(t) = current_bubbles(at_earlier_time(snapshot, t))
    M = Array{ComplexF64, 3}(undef, length(times), length(ks), length(tensor_directions))
    for (i, t) in enumerate(times)
        bs = _bubbles(t)
        M[i, :, :] .= T_ij(ks, bs, ϕ_resolution, μ_resolution, ΔV, tensor_directions; kwargs...)
    end 
    return M
end

function T_ij(ks:: Vector{Point3}, snapshot:: BubblesSnapShot, times:: Vector{Float64}, 
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector, Nothing} = nothing; 
              kwargs...):: Array{ComplexF64, 3}
    return T_ij(ks, snapshot, times, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions; kwargs...)
end

export T_ij

end