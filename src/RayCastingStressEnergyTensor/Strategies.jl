"""
    RayCastingSphericalQuadrature

Strategy struct for ray-casting T_ij computation with CosineWeight temporal decorrelation.

# Fields
- `quadrature::SphericalQuadratureScheme`: Spherical quadrature generator
- `markers::Vector{SphericalQuadratureMarker}`: Pre-computed ray markers (cached)
"""
struct RayCastingSphericalQuadrature
    quadrature::SphericalQuadratureScheme
    markers::Vector{SphericalQuadratureMarker}
end

function RayCastingSphericalQuadrature(quadrature::SphericalQuadratureScheme)
    return RayCastingSphericalQuadrature(quadrature, get_markers(quadrature))
end
