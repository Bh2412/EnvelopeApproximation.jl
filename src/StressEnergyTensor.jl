module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubblesIntegration
using EnvelopeApproximation.BubbleBasics
using Base.Iterators
import Meshes: Vec, Point3, coordinates, ⋅
import EnvelopeApproximation.BubblesIntegration.SurfaceIntegration: surface_integral, BubblePoint
using LinearAlgebra

TensorDirection{N} = Union{Symbol, NTuple{N, Symbol}} where N

function td_integrand(tensor_direction:: T):: Function where T <: TensorDirection
    if tensor_direction ≡ :trace
        return (p:: BubblePoint -> 1.)
    end
    CARTESIAN_DIRECTIONS = [:x, :y, :z]
    try
        indices:: Vector{Int64} = indexin(tensor_direction, CARTESIAN_DIRECTIONS)
        return (p:: BubblePoint -> prod(coordinates(p)[indices]))
    catch e
        if e isa MethodError
            throw(e("All directions must be elements of $CARTESIAN_DIRECTIONS"))
        end
    end
end

⋅(p1:: BubblePoint, p2:: Point3) = ⋅(coordinates.([p1, p2])...)

_exp(p:: BubblePoint, k:: Point3) = exp(-im * (p ⋅ k))


function surface_integrand(ks:: Vector{Point3}, tensor_directions:: Vector, 
                           tv_energy:: Float64 = 1.)
    ks = reshape(ks, (length(ks), 1))
    _td_integrand = reshape(td_integrand.(tensor_directions), (1, length(tensor_directions)))
    function _integrand(p:: BubblePoint):: Array{Complex, 2}
        return @. _exp((p, ), ks) * ((p, ) |> _td_integrand) * tv_energy
    end
    return _integrand
end


function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          tv_energy:: Float64 = 1.)
    integrand = surface_integrand(ks, tensor_directions, tv_energy)
    return surface_integral(integrand, bubbles, ϕ_resolution, μ_resolution)
end

function surface_integral(ks:: Vector{Point3}, 
    bubbles:: Bubbles, 
    tensor_directions:: Vector,
    n_ϕ:: Int64, 
    n_μ:: Int64,
    tv_energy:: Float64 = 1.)
    return surface_integral(ks, bubbles, tensor_directions, 
                            2π / n_ϕ, 2. / n_μ, tv_energy)
end

end