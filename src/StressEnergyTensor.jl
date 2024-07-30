module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.BubbleBasics
using Base.Iterators
import Meshes: Vec, Point3, coordinates, ⋅, -
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration: surface_integral, BubblePoint
import EnvelopeApproximation.BubblesIntegration.VolumeIntegration: volume_integral
using LinearAlgebra

TensorDirection{N} = Union{Symbol, NTuple{N, Symbol}} where N

-(p1:: BubblePoint, p2:: Point3) = Point3((p1.point - p2)...)

function td_integrand(tensor_direction:: T, bubbles:: Bubbles):: Function where T <: TensorDirection
    if tensor_direction ≡ :trace
        return (p:: BubblePoint -> 1.)
    end
    CARTESIAN_DIRECTIONS = [:x, :y, :z]
    try
        indices:: Vector{Int64} = indexin(tensor_direction, CARTESIAN_DIRECTIONS)
        N = length(indices)
        return (p:: BubblePoint -> prod(coordinates(p - bubbles[p.bubble_index].center)[indices]) / (bubbles[p.bubble_index].radius ^ N))
    catch e
        if e isa MethodError
            throw(e("All directions must be elements of $CARTESIAN_DIRECTIONS"))
        end
    end
end

⋅(p1:: BubblePoint, p2:: Point3) = ⋅(coordinates.([p1, p2])...)

_exp(p:: BubblePoint, k:: Point3) = exp(-im * (p ⋅ k))


function surface_integrand(ks:: Vector{Point3}, bubbles:: Bubbles, tensor_directions:: Vector, 
                           ΔV:: Float64 = 1.)
    ks = reshape(ks, (length(ks), 1))
    _td_integrand = reshape(td_integrand.(tensor_directions, (bubbles, )), (1, length(tensor_directions)))
    function _integrand(p:: BubblePoint):: Array{Complex, 2}
        return @. _exp((p, ), ks) * ((p, ) |> _td_integrand) * (ΔV * bubbles[p.bubble_index].radius / 3)
    end
    return _integrand
end


function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          ΔV:: Float64 = 1.):: Array{ComplexF64, 2}
    integrand = surface_integrand(ks, bubbles, tensor_directions, ΔV)
    return surface_integral(integrand, bubbles, ϕ_resolution, μ_resolution)
end

function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          n_ϕ:: Int64, 
                          n_μ:: Int64,
                          ΔV:: Float64 = 1.)
    return surface_integral(ks, bubbles, tensor_directions, 
                            2π / n_ϕ, 2. / n_μ, ΔV)
end

export surface_integral

function volume_integrand(ks:: Vector{Point3}, ΔV:: Float64 = 1.)
    function integrand(p:: BubblePoint):: Vector{ComplexF64}
        return @. _exp((p, ), ks) * ΔV
    end
end

function volume_integral(ks:: Vector{Point3}, bubbles:: Bubbles, 
                         v_resolution:: Float64, 
                         ϕ_resolution:: Float64, 
                         μ_resolution:: Float64, 
                         ΔV:: Float64 = 1.) 
    return volume_integral(volume_integrand(ks, ΔV), bubbles, v_resolution, ϕ_resolution, μ_resolution)
end

function volume_integral(ks:: Vector{Point3}, bubbles:: Bubbles, 
                         n_v:: Int64, 
                         n_ϕ:: Int64, 
                         n_μ:: Int64, 
                         ΔV:: Float64 = 1.)
    return volume_integral(ks, bubbles, (1. / 3) / n_v, 2π / n_ϕ, 2. / n_μ, ΔV)
end

export volume_integral

P(T_ii:: Array{ComplexF64}, V:: Array{ComplexF64}) = @.((1. / 3) * T_ii - V)

ρ(T_ii:: Array{ComplexF64}, V:: Array{ComplexF64}) = @.(T_ii + V)

CARTESIAN_DIRECTIONS = [:x, :y, :z]
diagonal = [(s, s) for s in CARTESIAN_DIRECTIONS]
above_diagonal = [(:x, :y), (:x, :z), (:y, :z)]
upper_right = vcat(diagonal, above_diagonal)

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

function T_ij(ks:: Vector{Point3}, 
              bubbles:: Bubbles, 
              v_resolution:: Float64,
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{TensorDirection, Nothing} = nothing):: Dict{Union{TensorDirection, Symbol}, Union{Vector{ComplexF64}, Vector{Point3}}}
    isnothing(tensor_directions) && (tensor_directions = vcat([:trace], upper_right))
    si = surface_integral(ks, bubbles, tensor_directions, ϕ_resolution,
                          μ_resolution, ΔV)
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
        vi = volume_integral(ks, bubbles, v_resolution, ϕ_resolution, 
                             μ_resolution, ΔV)
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
              n_v:: Int64,
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1., 
              tensor_directions:: Union{TensorDirection, Nothing} = nothing):: Dict{TensorDirection, Union{Vector{ComplexF64}, Vector{Point3}}}
    return T_ij(ks, bubbles, (1. / 3) / n_v, 2π / n_ϕ, 2. / n_μ, ΔV, tensor_directions)
end


function state_parameters(ks:: Vector{Point3}, 
              bubbles:: Bubbles, 
              v_resolution:: Float64,
              ϕ_resolution:: Float64,
              μ_resolution:: Float64,
              ΔV:: Float64 = 1.):: Dict{Symbol, Union{Vector{ComplexF64}, Vector{Point3}}}
    tds = vcat([:trace], upper_right)
    si = surface_integral(ks, bubbles, tds, ϕ_resolution,
                          μ_resolution, ΔV)
    V = volume_integral(ks, bubbles, v_resolution, ϕ_resolution, 
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
              n_v:: Int64,
              n_ϕ:: Int64,
              n_μ:: Int64,
              ΔV:: Float64 = 1.):: Dict{Symbol, Union{Vector{ComplexF64}, Vector{Point3}}}
    return state_parameters(ks, bubbles, (1. / 3) / n_v, 2π / n_ϕ, 2. / n_μ, ΔV)
end

export state_parameters

end