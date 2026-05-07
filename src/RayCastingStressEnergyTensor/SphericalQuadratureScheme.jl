using StaticArrays

"""
    SphericalQuadratureMarker

Represents a single ray direction with associated quadrature weight.

# Fields
- `n̂::SVector{3, Float64}`: Unit direction vector of the ray
- `weight::Float64`: Quadrature weight w_a
"""
struct SphericalQuadratureMarker
    n̂::SVector{3, Float64}
    weight::Float64
end

"""
    abstract type SphericalQuadratureScheme

Interface for spherical quadrature schemes. Implementations must provide:
  `get_markers(scheme::T)::Vector{SphericalQuadratureMarker}`
"""
abstract type SphericalQuadratureScheme end

"""
    UniformSphericalCapScheme <: SphericalQuadratureScheme

Simple uniform spherical cap quadrature: N_θ rings × N_ϕ azimuthal points.

# Fields
- `N_theta::Int`: Number of polar angle points
- `N_phi::Int`: Number of azimuthal points per ring
"""
struct UniformSphericalCapScheme <: SphericalQuadratureScheme
    N_theta::Int
    N_phi::Int
end

function get_markers(scheme::UniformSphericalCapScheme)::Vector{SphericalQuadratureMarker}
    N_θ, N_ϕ = scheme.N_theta, scheme.N_phi

    markers = SphericalQuadratureMarker[]

    # Weight for unit sphere quadrature: each cell contributes Δμ × Δϕ

    Δμ = 2.0 / N_θ
    Δϕ = 2.0 * π / N_ϕ
    w = Δμ * Δϕ

    # Polar angles: cos(θ) ∈ [-1, 1] uniform in μ = cos(θ)
    for i in 1:N_θ

        μ = i * Δμ - 1.0 - Δμ / 2
        s  = sqrt(clamp(1.0 - μ^2, 0.0, 1.0))  # sin(θ) for Cartesian conversion
        # Azimuthal angles
        for j in 1:N_ϕ
            ϕ = j * Δϕ - Δϕ / 2  # Centered in each azimuthal cell

            n̂ = SVector(s * cos(ϕ), s * sin(ϕ), μ)
            push!(markers, SphericalQuadratureMarker(n̂, w))
        end
    end

    return markers
end