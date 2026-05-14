"""
    RayCastingSphericalQuadrature

Strategy struct for ray-casting T_ij computation with CosineWeight temporal decorrelation.

# Fields
- `quadrature::SphericalQuadratureScheme`: Spherical quadrature generator
"""
struct RayCastingSphericalQuadrature
    quadrature::SphericalQuadratureScheme
end

function RayCastingSphericalQuadrature(quadrature::SphericalQuadratureScheme)
    return RayCastingSphericalQuadrature(quadrature)
end
