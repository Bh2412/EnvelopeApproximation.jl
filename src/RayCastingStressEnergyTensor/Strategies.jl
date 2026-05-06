"""
    RayCastingT_ij_CosineWeight

Strategy struct for ray-casting T_ij computation with CosineWeight temporal decorrelation.

# Fields
- `quadrature::SphericalQuadratureScheme`: Spherical quadrature generator
- `markers::Vector{SphericalQuadratureMarker}`: Pre-computed ray markers (cached)
"""
struct RayCastingT_ij_CosineWeight
    quadrature::SphericalQuadratureScheme
    markers::Vector{SphericalQuadratureMarker}
end

function RayCastingT_ij_CosineWeight(quadrature::SphericalQuadratureScheme)
    return RayCastingT_ij_CosineWeight(quadrature, get_markers(quadrature))
end
