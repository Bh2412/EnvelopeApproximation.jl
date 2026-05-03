using QuadGK
using Bessels
using StaticArrays
using LinearAlgebra

"""
    compute_time_domain_Tij(k_vec, r1, r2, d, ρ_vac)

Computes the generalized T̃_ij(k) for two bubbles of DIFFERENT sizes r1, r2.
Includes the potential contribution as a 5th element of the surface integral, 
preserving the existing kinetic/gradient terms.
"""
function compute_time_domain_Tij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, ρ_vac::T=1.0)::SVector{6, Complex{T}} where T<:Real
    # 1. Setup Geometry
    kx = sqrt(k_vec[1]^2 + k_vec[2]^2)
    kz = k_vec[3]
    k2 = kx^2 + kz^2 # k^2 needed for the potential prefactor

    # Calculate Intersection Angles (Law of Cosines)
    if d >= r1 + r2
        alpha1 = zero(T)
        alpha2 = zero(T)
    elseif d <= abs(r1 - r2)
        alpha1 = zero(T)
        alpha2 = zero(T)
    else
        cos_a1 = clamp((r1^2 + d^2 - r2^2) / (2 * r1 * d), -one(T), one(T))
        cos_a2 = clamp((r2^2 + d^2 - r1^2) / (2 * r2 * d), -one(T), one(T))
        alpha1 = acos(cos_a1)
        alpha2 = acos(cos_a2)
    end

    # 2. Integration Helper
    # Computes a 5-element vector: 4 kinetic components + 1 potential integral core
    function integrate_bubble(r, theta_min, theta_max)
        function integrand(theta)
            s, c = sincos(theta)
            u = kx * r * s
            phase = exp(-im * kz * r * c)
            
            j0 = besselj0(u)
            j1 = besselj1(u)
            j2 = besselj(2, u)
            
            # First 4: Unaffected Kinetic Integrands
            val_xx = s^3 * (j0 - j2) * phase
            val_yy = s^3 * (j0 + j2) * phase
            val_zz = 2 * s * c^2 * j0 * phase
            val_xz = -2im * s^2 * c * j1 * phase
            
            # 5th Element: Potential Integrand (integrating θ and incorporating azimuthal Bessel expansion)
            val_pot_int = s * (kz * c * j0 - im * kx * s * j1) * phase
            
            return SVector{5, Complex{T}}(val_xx, val_yy, val_zz, val_xz, val_pot_int)
        end
        
        res, _ = quadgk(integrand, theta_min, theta_max)
        return res
    end

    # 3. Compute Contributions with Limits
    # Bubble 1 (Left, z = -d/2)
    phase_shift1 = exp(im * kz * d/2)
    I1 = integrate_bubble(r1, alpha1, π)
    
    # Bubble 2 (Right, z = +d/2)
    phase_shift2 = exp(-im * kz * d/2)
    I2 = integrate_bubble(r2, zero(T), π - alpha2)

    # 4. Prefactors
    # Kinetic prefactors (kept as originally defined in your code)
    factor1_kin = (π * ρ_vac / 3) * r1^3 * phase_shift1
    factor2_kin = (π * ρ_vac / 3) * r2^3 * phase_shift2

    # Potential prefactors: 2πi * ρ_vac * R^2 / k^2 * phase_shift
    factor1_pot = (r1 > 0 && k2 > 0) ? (2 * π * im * ρ_vac / k2) * r1^2 * phase_shift1 : zero(Complex{T})
    factor2_pot = (r2 > 0 && k2 > 0) ? (2 * π * im * ρ_vac / k2) * r2^2 * phase_shift2 : zero(Complex{T})

    # 5. Sum and Normalize
    Txx = factor1_kin * I1[1] + factor1_pot * I1[5] + factor2_kin * I2[1] + factor2_pot * I2[5]
    Tyy = factor1_kin * I1[2] + factor1_pot * I1[5] + factor2_kin * I2[2] + factor2_pot * I2[5]
    Tzz = factor1_kin * I1[3] + factor1_pot * I1[5] + factor2_kin * I2[3] + factor2_pot * I2[5]
    Txz = factor1_kin * I1[4] + factor2_kin * I2[4] # Txz cross-term receives no potential contribution

    return SVector{6, Complex{T}}(Txx, 0., Txz, Tyy, 0., Tzz)
end

function AnalyticTwoPointTij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, ρ_vac::T=1.0, V:: T = 1.)::SMatrix{6,6,Complex{T}} where T<:Real
    Tij = compute_time_domain_Tij(k_vec, r1, r2, d, ρ_vac)
    return (Tij * Tij') ./ V
end
