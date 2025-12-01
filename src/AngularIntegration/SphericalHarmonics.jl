module SphericalHarmonics

using FastTransforms
import EnvelopeApproximation.AngularIntegrationInterface: AbstractAngularIntegrationPlan, integrate_angular

export spherical_harmonic_coefficients, monopole, SHPlan, integrate_angular

"""
Compute spherical harmonic coefficients of a vector-valued function f(ϕ, θ). The output matrix follows the convention:

Matrix layout (showing which Yℓᵐ is at each position):

           col=1   col=2   col=3   col=4   col=5   col=6   col=7   ...
row=1  │   Y₀⁰    Y₁⁻¹    Y₁¹     Y₂⁻²    Y₂²     Y₃⁻³    Y₃³    ...
row=2  │   Y₁⁰    Y₂⁻¹    Y₂¹     Y₃⁻²    Y₃²     Y₄⁻³    Y₄³    ...
row=3  │   Y₂⁰    Y₃⁻¹    Y₃¹     Y₄⁻²    Y₄²     Y₅⁻³    Y₅³    ...
row=4  │   Y₃⁰    Y₄⁻¹    Y₄¹     Y₅⁻²    Y₅²     Y₆⁻³    Y₆³    ...
  ⋮    │    ⋮       ⋮       ⋮       ⋮       ⋮       ⋮       ⋮     ⋱

Arguments:
- f: Function taking (ϕ, θ) and returning a K-dimensional vector/array
- lmax: Maximum spherical harmonic degree
- K: Dimension of output of f

Returns:
- sh_coeffs: Array of spherical harmonic coefficients with size (lmax+1) × (2*lmax+1) × K
"""
function spherical_harmonic_coefficients(f, lmax::Int, K::Int)
    # Grid dimensions
    m = lmax + 1          # number of θ points
    n = 2 * lmax + 1      # number of ϕ points (must be odd)
    
    # Create grid
    θs = map(k -> π * (k - 0.5) / m, 1:m)
    ϕs = map(j -> 2π * (j - 1) / n, 1:n)
    
    # Sample vector-valued function on grid
    grid = zeros(ComplexF64, m, n, K)
    for i in 1:m, j in 1:n
        @views grid[i, j, :] = f(ϕs[j], θs[i])
    end
    
    # Compute SH coefficients for each component
    sh_coeffs = zeros(ComplexF64, m, n, K)
    
    # Create plans once (same for all components)
    plan_analysis = plan_sph_analysis(grid[:, :, 1])
    plan_sph = plan_sph2fourier(grid[:, :, 1])
    
    for k in 1:K
        # Transform each component independently
        @views fourier_coeffs = plan_analysis * grid[:, :, k]
        @views sh_coeffs[:, :, k] = plan_sph \ fourier_coeffs
    end
    
    return sh_coeffs
end

"""
    monopole(sh_coeffs)

Compute the integral of the vector field over the sphere using the monopole moment (Y₀⁰) 
from pre-calculated coefficients.
"""
function monopole(sh_coeffs::AbstractArray{ComplexF64, 3})
    # The integral of Y₀⁰ over the unit sphere is √4π.
    # In FastTransforms layout, the (0,0) mode is at index [1, 1].
    return sh_coeffs[1, 1, :] .* sqrt(4π)
end

"""
    monopole(f, lmax, K)

Compute the spherical harmonic coefficients of `f` up to `lmax` and return the monopole integral.
"""
function monopole(f, lmax::Int, K::Int)
    coeffs = spherical_harmonic_coefficients(f, lmax, K)
    return monopole(coeffs)
end

"""
    SHPlan

Integration strategy using Spherical Harmonics decomposition.
This is efficient for smooth functions and allows for easy spectral analysis.

Fields:
- `lmax`: The maximum spherical harmonic degree (resolution).
- `K`: The dimension of the output vector (e.g., number of k-modes/wavenumbers).
"""
struct SHPlan <: AbstractAngularIntegrationPlan
    lmax::Int
    K::Int
end

function integrate_angular(plan::SHPlan, f_directional::Function)
    # 1. Decompose the directional function into spherical harmonics [cite: 47]
    coeffs = spherical_harmonic_coefficients(f_directional, plan.lmax, plan.K)
    
    # 2. The integral over the sphere is exactly the monopole moment * sqrt(4π)
    # The 'monopole' function in SphericalHarmonics.jl already applies this factor[cite: 51].
    return monopole(coeffs)
end

end