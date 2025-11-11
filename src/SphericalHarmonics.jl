module SphericalHarmonics

using FastTransforms

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

export spherical_harmonic_coefficients

end