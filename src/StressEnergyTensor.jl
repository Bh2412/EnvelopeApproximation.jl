module StressEnergyTensor
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.BubbleBasics: ⋅
using EnvelopeApproximation.BubblesEvolution
using Base.Iterators
import Base./
import EnvelopeApproximation.SurfaceIntegration: surface_sections, BubbleSection, unit_sphere_point, bubble_point
using StaticArrays
using HCubature
import HCubature: hcubature
using Tullio

TensorDirection = Union{Symbol, Tuple{Symbol, Symbol}}

export TensorDirection

unit_sphere_point(x:: SVector{2, Float64}) = unit_sphere_point(x...)

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

function td_integrand(x:: SVector{2, Float64}, tensor_direction:: Tuple{Symbol, Symbol}):: Float64 
    return td_integrand(x, tensor_direction[1]) * td_integrand(x, tensor_direction[2])
end

function td_integrand(x:: SVector{2, Float64}, tensor_directions:: Matrix):: Matrix{Float64}
    return td_integrand.((x, ), tensor_directions)
end

⋅(p1:: Point3, p2:: Point3):: Float64 = coordinates(p1) ⋅ coordinates(p2)
/(p:: Point3, d:: Float64):: Point3 = Point3((coordinates(p) / d)...)

function _exp(p:: Point3, k:: Point3):: SVector{2, Float64}
    d = p ⋅ k
    return SVector{2, Float64}(cos(d), -sin(d))
end

function coordinate_transformation(x:: SVector{2, Float64}, s:: BubbleSection):: SVector{2, Float64}
    """
    The integration over each section is performed between (0., 2π), (-1., 1.), this transformation makes sure of that
    """
    return @. (x * (s.ϕ.d / 2π, s.μ.d / 2)) + (s.ϕ.c + s.ϕ.d / 2 - π, s.μ.c)
end

measure(s:: BubbleSection) = s.ϕ.d * s.μ.d

function add_section_contribution!(V:: Array{Float64, 3}, x:: SVector{2, Float64}, s:: BubbleSection, ks:: Vector{Point3}, 
                                   bubble:: Bubble, tensor_directions:: Vector{TensorDirection}, ΔV:: Float64)
    px = coordinate_transformation(x, s)
    p = bubble_point(px..., bubble)
    c = (bubble.radius ^ 3 * ((ΔV / 3.) * measure(s) / 4π))
    @inbounds for (l, td) ∈ enumerate(tensor_directions), (j, k) ∈ enumerate(ks), i ∈ 1:2
        V[i, j, l] += _exp(p, k)[i] * td_integrand(px, td) * c
    end
end

function surface_integrand(x:: SVector{2, Float64}, surface_sections:: Vector{BubbleSection}, ks:: Vector{Point3}, 
                           bubbles:: Bubbles, tensor_directions:: Vector{TensorDirection}, ΔV:: Float64):: Array{Float64, 3}
    V = zeros(Float64, 2, length(ks), length(tensor_directions))
    for section in surface_sections
        add_section_contribution!(V, x, section, ks, bubbles[section.bubble_index], tensor_directions, ΔV)
    end
    return V
end

function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector{TensorDirection},
                          ϕ_resolution:: Float64,
                          μ_resolution:: Float64,
                          ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    sections = surface_sections(ϕ_resolution, μ_resolution, bubbles)
    integrand(x:: SVector{2, Float64}):: Array{Float64, 3} = surface_integrand(x, sections, ks, bubbles, tensor_directions, ΔV)
    x0, xf = SVector{2, Float64}(0., -1.), SVector{2, Float64}(2π, 1.)
    V = hcubature(integrand, x0, xf; kwargs...)[1]
    return reshape(reinterpret(ComplexF64, V), length(ks), length(tensor_directions))
end

function surface_integral(ks:: Vector{Point3}, 
                          bubbles:: Bubbles, 
                          tensor_directions:: Vector,
                          n_ϕ:: Int64, 
                          n_μ:: Int64,
                          ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
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