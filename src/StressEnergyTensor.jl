module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.BubbleBasics
using Base.Iterators
import Base./
import Meshes: Vec, Point3, coordinates, ⋅, -
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration: surface_integral, BubbleSection, unit_sphere_point, bubble_point
using LinearAlgebra
using StaticArrays
using HCubature
import HCubature: hcubature

TensorDirection{N} = Union{Symbol, NTuple{N, Symbol}} where N

function td_integrand(x:: SVector{2, Float64}, tensor_direction):: Float64 
    if tensor_direction ≡ :trace
        return 1.
    end
    CARTESIAN_DIRECTIONS = [:x, :y, :z]
    try
        indices:: Vector{Int64} = indexin(tensor_direction, CARTESIAN_DIRECTIONS)
        return prod(coordinates(unit_sphere_point(x...))[indices])
    catch e
        if e isa MethodError
            throw(e("All directions must be elements of $CARTESIAN_DIRECTIONS"))
        end
    end
end

function td_integrand(x:: SVector{2, Float64}, tensor_directions:: Vector):: Vector{Float64}
    return td_integrand.((x, ), tensor_directions)
end

⋅(p1:: Point3, p2:: Point3):: Float64 = ⋅(coordinates.([p1, p2])...)
⋅(p1:: BubbleSection, p2:: Point3):: Float64 = ⋅(coordinates.([p1, p2])...)
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
    return unit_sphere_point(x...) .⋅ ks
end

function potential_integrand(ks:: Vector{Point3}, bubbles:: Bubbles,
                             ΔV:: Float64 = 1.)
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

P(T_ii:: Array{ComplexF64}, V:: Array{ComplexF64}) = @.((1. / 3) * T_ii - V)

ρ(T_ii:: Array{ComplexF64}, V:: Array{ComplexF64}) = @.(T_ii + V)

CARTESIAN_DIRECTIONS = [:x, :y, :z]
diagonal = [(s, s) for s in CARTESIAN_DIRECTIONS]
above_diagonal = [(:x, :y), (:x, :z), (:y, :z)]
upper_right = vcat(diagonal, above_diagonal)

function T_ij(ks:: Vector{Point3}, 
              bubbles:: Bubbles, 
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{Vector{TensorDirection}, Nothing} = nothing; 
              kwargs...):: Dict{Union{TensorDirection, Symbol}, Union{Vector{ComplexF64}, Vector{Point3}}}
    isnothing(tensor_directions) && (tensor_directions = vcat([:trace], upper_right))
    si = surface_integral(ks, bubbles, tensor_directions, ϕ_resolution,
                          μ_resolution, ΔV; kwargs...)
    T = Dict{Union{TensorDirection, Symbol}, Union{Vector{ComplexF64}, Vector{Point3}}}()
    T[:k] = ks
    for (i, td) in enumerate(tensor_directions)
        T[td] = reshape(si[:, i], length(ks))
    end
     # Based on the following:
    ```math
    T_ij = ∂_iϕ∂_jϕ - δ_ij⋅L ≈ ∂_iϕ∂_jϕ - δ_ij V
    ```         
    if any(td in tensor_directions for td in diagonal)
        vi = potential_integral(ks, bubbles, ϕ_resolution, 
                                μ_resolution, ΔV; kwargs...)
        for td in tensor_directions
            if td in diagonal
                T[td] -= vi
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
              tensor_directions:: Union{Vector{TensorDirection}, Nothing} = nothing; 
              kwargs...):: Dict{TensorDirection, Union{Vector{ComplexF64}, Vector{Point3}}}
    return T_ij(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions; kwargs...)
end

export T_ij

function auto_outer_product(ks:: Vector{Point3}, td:: TensorDirection):: Vector{Float64}
    indices = indexin(td, [:x, :y, :z])
    return map(k -> prod(coordinates(k)[indices]), ks)
end

function auto_outer_product(ks:: Vector{Point3}, tds:: Vector):: Array{Float64, 2}
    return map(td -> auto_outer_product(ks, td), tds) |> x -> hcat(x...)
end

function N_φ(ks:: Vector{Point3}, Π_μν:: Array{ComplexF64, 2}, 
             tds:: Vector):: Vector{ComplexF64}
    aop = auto_outer_product(ks, tds)
    res = @. (-3. / 2) * Π_μν * aop
    adi = indexin(above_diagonal, tds)
    di = indexin(diagonal, tds)
    res = 2 .* (sum(res[:, adi], dims=2)) .+ sum(res[:, di], dims=2)
    return reshape(res, length(ks))
end

function state_parameters(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          ΔV:: Float64 = 1.):: Dict{Symbol, Union{Vector{ComplexF64}, Vector{Point3}}}
    tds = vcat([:trace], upper_right)
    si = surface_integral(ks, bubbles, tds, ϕ_resolution,
                          μ_resolution, ΔV)
    V = potential_integral(ks, bubbles, ϕ_resolution, 
                           μ_resolution, ΔV)
    T_ii = reshape(si[:, 1], length(ks))
    T = Dict{Symbol, Union{Vector{ComplexF64}, Vector{Point3}}}()
    T[:k] = ks
    T[:ρ] = ρ(T_ii, V)
    T[:P] = P(T_ii, V)
    Π_μν, _tds = begin
        t = indexin((:trace, ), tds)
        d = @. si[:, $indexin(diagonal, tds)] - (1. / 3) * si[:, t]
        ad = si[:, indexin(above_diagonal, tds)]
        hcat(d, ad), vcat(diagonal, above_diagonal)
    end
    T[:N_φ] = N_φ(ks, Π_μν, _tds)  
    return T
end

export state_parameters
   
function state_parameters(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          n_ϕ:: Int64,
                          n_μ:: Int64,
                          ΔV:: Float64 = 1.):: Dict{Symbol, Union{Vector{ComplexF64}, Vector{Point3}}}
    return state_parameters(ks, bubbles, 2π / n_ϕ, 2. / n_μ, ΔV)
end

export state_parameters

end