using QuadGK
using Bessels
using StaticArrays
using LinearAlgebra

"""
    compute_time_domain_Tij(k_vec, r, d, ρ_vac)

Computes the instantaneous spatial Fourier transform of the stress-energy tensor 
T̃_ij(k, r) for a two-bubble system in the envelope approximation.

Arguments:
- `k_vec`: Wavevector [kx, ky, kz]. The function assumes azimuthal symmetry 
           about z, so kx is taken as the magnitude in the xy-plane.
- `r`:     Current bubble radius (equivalent to time t since v_wall ≈ 1).
- `d`:     Separation distance between bubble centers.
- `ρ_vac`: Vacuum energy density (default: 1.0).

Returns:
- A tuple (Txx, Tyy, Tzz, Txz) representing the components.
"""
function compute_time_domain_Tij(k_vec::AbstractVector{T}, r::T, d::T, ρ_vac::T=1.0):: SVector{4, T} where T<:Real
    # 1. Setup Geometry
    # -----------------
    # k_vec decomposition (project to x-z plane due to cylindrical symmetry)
    kx = sqrt(k_vec[1]^2 + k_vec[2]^2)
    kz = k_vec[3]

    # Collision angle alpha
    # cos(alpha) = d / (2r) for overlapping bubbles
    if r < d/2
        alpha = zero(T)
    else
        val = d / (2*r)
        val = clamp(val, -one(T), one(T)) # Numerical safety
        alpha = acos(val)
    end

    # 2. Define Vector-Valued Integrand
    # ---------------------------------
    function integrand(theta)
        s, c = sincos(theta)
        s2 = s^2
        s3 = s^3
        c2 = c^2
        
        # Arguments for Bessel and Trig functions
        u = kx * r * s
        phase_arg = kz * r * c + kz * d / 2
        
        # Compute terms
        j0 = besselj0(u)
        j1 = besselj1(u)
        j2 = besselj(2, u)
        cos_phase = cos(phase_arg)
        sin_phase = sin(phase_arg)

        # Raw integral components (before prefactors)
        # We group terms to minimize function calls
        # Component 1 (for Txx): sin^3 * cos(phase) * (J0 - J2)
        val_1 = s3 * cos_phase * (j0 - j2)
        
        # Component 2 (for Tyy): sin^3 * cos(phase) * (J0 + J2)
        val_2 = s3 * cos_phase * (j0 + j2)
        
        # Component 3 (for Tzz): sin * cos^2 * cos(phase) * J0
        val_3 = s * c2 * cos_phase * j0
        
        # Component 4 (for Txz): sin^2 * cos * sin(phase) * J1
        val_4 = s2 * c * sin_phase * j1
        
        return SVector{4, T}(val_1, val_2, val_3, val_4)
    end

    # 3. Perform Integration
    # ----------------------
    # Integrate from 0 to pi - alpha
    upper_limit = π - alpha
    results, error = quadgk(integrand, zero(T), upper_limit)

    # 4. Apply Prefactors and Return
    # ------------------------------
    # Base factor: 2π * ρ * r^3 / 3
    prefactor_base = (2 * π * ρ_vac / 3) * r^3

    Txx = prefactor_base * results[1]
    Tyy = prefactor_base * results[2]
    # Tzz has a 4π factor (2 * base)
    Tzz = 2 * prefactor_base * results[3]
    # Txz has a -4π factor (-2 * base)
    Txz = -2 * prefactor_base * results[4]

    return SVector{4, T}(Txx, Tyy, Tzz, Txz)
end

"""
    compute_time_domain_Tij_general(k_vec, r1, r2, d, ρ_vac)

Computes the generalized T̃_ij(k) for two bubbles of DIFFERENT sizes r1, r2.
Correctly handles the integration limits for the left (r1) and right (r2) bubbles.
"""
function compute_time_domain_Tij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, ρ_vac::T=1.0):: SVector{4, Complex{T}}where T<:Real
    # 1. Setup Geometry
    kx = sqrt(k_vec[1]^2 + k_vec[2]^2)
    kz = k_vec[3]

    # Calculate Intersection Angles (Law of Cosines)
    if d >= r1 + r2
        # No overlap -> Full spheres
        alpha1 = zero(T)
        alpha2 = zero(T)
    elseif d <= abs(r1 - r2)
        # One inside the other -> Nested (Usually no radiation in Envelope Approx, but we calculate shape)
        # If r1 > r2, bubble 2 is inside. Bubble 1 is full? 
        # Standard envelope approximation implies only the outermost surface radiates.
        # We'll assume standard partial overlap geometry for limits.
        alpha1 = zero(T)
        alpha2 = zero(T)
    else
        # Partial Overlap
        cos_a1 = (r1^2 + d^2 - r2^2) / (2 * r1 * d)
        cos_a2 = (r2^2 + d^2 - r1^2) / (2 * r2 * d)
        
        # Clamp for numerical safety
        alpha1 = acos(clamp(cos_a1, -one(T), one(T)))
        alpha2 = acos(clamp(cos_a2, -one(T), one(T)))
    end

    # 2. Integration Helper
    # Computes the integral vector for a single bubble
    function integrate_bubble(r, theta_min, theta_max)
        function integrand(theta)
            s, c = sincos(theta)
            u = kx * r * s
            
            # Phase from surface position relative to bubble center
            # exp(-i * kz * r * cos(theta))
            phase = exp(-im * kz * r * c)
            
            j0 = besselj0(u)
            j1 = besselj1(u)
            j2 = besselj(2, u)
            
            # Txx components (J0 - J2)
            val_xx = s^3 * (j0 - j2) * phase
            # Tyy components (J0 + J2)
            val_yy = s^3 * (j0 + j2) * phase
            # Tzz components (2 * J0)
            val_zz = 2 * s * c^2 * j0 * phase
            # Txz components (-2i * J1)
            val_xz = -2im * s^2 * c * j1 * phase
            
            return SVector{4, Complex{T}}(val_xx, val_yy, val_zz, val_xz)
        end
        
        res, _ = quadgk(integrand, theta_min, theta_max)
        return res
    end

    # 3. Compute Contributions with CORRECT LIMITS
    # --------------------------------------------
    
    # Bubble 1 (Left, z = -d/2)
    # Collision is at theta = 0 (forward). Uncollided is BACK.
    # Limit: alpha1 -> pi
    phase_shift1 = exp(im * kz * d/2)
    I1 = integrate_bubble(r1, alpha1, π)
    
    # Bubble 2 (Right, z = +d/2)
    # Collision is at theta = pi (backward). Uncollided is FRONT.
    # Limit: 0 -> pi - alpha2
    phase_shift2 = exp(-im * kz * d/2)
    I2 = integrate_bubble(r2, zero(T), π - alpha2)

    # 4. Sum and Normalize
    # Prefactor: (rho / 3) * R^3
    factor1 = (π * ρ_vac / 3) * r1^3 * phase_shift1
    factor2 = (π * ρ_vac / 3) * r2^3 * phase_shift2
    
    T_vec = factor1 .* I1 .+ factor2 .* I2
    
    return T_vec # (Txx, Tyy, Tzz, Txz)
end

"""
    compute_time_domain_Π(Tij::SVector{4, T}, ξ::T) where T<:Real

Computes the contraction Π(t, k) for a specific angle ξ.
Since the two-bubble system (axisymmetric) yields purely Real stress tensor components,
the scalar amplitude A is Real.

Formula: Π = (1 / 2V) * A^2
Where A = Tzz*sin²ξ + Txx*cos²ξ - Tyy - 2*Txz*sinξ*cosξ
"""
function compute_time_domain_Π(Tij::SVector{4, T}, ξ::T, V:: T) where T<:Real
    # 1. Unpack Stress Tensor Components (All Real)
    # ---------------------------------------------
    Txx = Tij[1]
    Tyy = Tij[2]
    Tzz = Tij[3]
    Txz = Tij[4] # This is the physical Real value

    # 2. Geometry Factors
    # -------------------
    s, c = sincos(ξ)
    s2 = s^2
    c2 = c^2
    sc = s * c

    # 3. Construct Scalar Amplitude A (Real)
    # --------------------------------------
    # A = Tzz*sin²ξ + Txx*cos²ξ - Tyy - 2*Txz*sinξ*cosξ
    A = Tzz * s2 + Txx * c2 - Tyy - 2 * Txz * sc

    # 4. Compute Contraction
    # ----------------------
    # Π = (1/2) * A^2
    return 0.5 * A^2 / V
end

"""
    compute_time_domain_Π(Tij::SVector{4, T}, ξ::T) where T<:Real

Computes the contraction Π(t, k) for a specific angle ξ.
Since the two-bubble system (axisymmetric) yields purely Real stress tensor components,
the scalar amplitude A is Real.

Formula: Π = (1 / 2V) * A^2
Where A = Tzz*sin²ξ + Txx*cos²ξ - Tyy - 2*Txz*sinξ*cosξ
"""
function compute_time_domain_Π(Tij::SVector{4, Complex{T}}, ξ::T, V:: T) where T<:Real
    # 1. Unpack Stress Tensor Components (All Real)
    # ---------------------------------------------
    Txx = Tij[1]
    Tyy = Tij[2]
    Tzz = Tij[3]
    Txz = Tij[4] # This is the physical Real value

    # 2. Geometry Factors
    # -------------------
    s, c = sincos(ξ)
    s2 = s^2
    c2 = c^2
    sc = s * c

    # 3. Construct Scalar Amplitude A (Real)
    # --------------------------------------
    # A = Tzz*sin²ξ + Txx*cos²ξ - Tyy - 2*Txz*sinξ*cosξ
    A = Tzz * s2 + Txx * c2 - Tyy - 2 * Txz * sc

    # 4. Compute Contraction
    # ----------------------
    # Π = (1/2) * A^2
    return 0.5 * abs2(A) / V
end

function compute_time_domain_Π(k_vec::AbstractVector{T}, r::T, d::T, ξ::T, ρ_vac::T=1.0, V=1.) where T<:Real
    Tij = compute_time_domain_Tij(k_vec, r, d, ρ_vac)
    return compute_time_domain_Π(Tij, ξ, V)
end

function compute_time_domain_Π(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, ξ::T, ρ_vac::T=1.0, V=1.) where T<:Real
    Tij = compute_time_domain_Tij(k_vec, r1, r2, d, ρ_vac)
    return compute_time_domain_Π(Tij, ξ, V)
end


"""
compute_analytic_isotropic_Pi(k, r, d, V)

Computes the Isotropic Power Spectral Density by integrating the direction-dependent
result over all solid angles.

Π_iso(k) = (1 / 4π) ∫ Π(k, Ω) dΩ  = (1 / 2) ∫_0^π Π(k, ξ) sin(ξ) dξ
"""
function compute_analytic_isotropic_Π(k_mag::Float64, r::Float64, d::Float64, V::Float64)
    
    # Define integrand: Π(k, ξ) * sin(ξ)
    function iso_integrand(xi)
        # Construct k_vec for this angle xi (assuming azimuthal symmetry, phi=0 is fine)
        # k lies in x-z plane
        k_vec = SVector(k_mag * sin(xi), 0.0, k_mag * cos(xi))
        
        # Compute Tensor Components
        Tij = compute_time_domain_Tij(k_vec, r, d, 1.0) # rho_vac = 1.0
        
        # Compute Π value for this direction
        Pi_val = compute_time_domain_Π(Tij, xi, V)
        
        # Return weighted by differential solid angle element
        return Pi_val * sin(xi)
    end

    # Integrate from 0 to π
    val, err = quadgk(iso_integrand, 0.0, π)
    
    # Divide by 2 (result of ∫ 1/2 d(cosθ))
    return val / 2.0
end

"""
compute_analytic_isotropic_Pi(k, r, d, V)

Computes the Isotropic Power Spectral Density by integrating the direction-dependent
result over all solid angles.

Π_iso(k) = (1 / 4π) ∫ Π(k, Ω) dΩ  = (1 / 2) ∫_0^π Π(k, ξ) sin(ξ) dξ
"""
function compute_analytic_isotropic_Π(k_mag::Float64, r1::Float64, r2::Float64, d::Float64, V::Float64)
    
    # Define integrand: Π(k, ξ) * sin(ξ)
    function iso_integrand(xi)
        # Construct k_vec for this angle xi (assuming azimuthal symmetry, phi=0 is fine)
        # k lies in x-z plane
        k_vec = SVector(k_mag * sin(xi), 0.0, k_mag * cos(xi))
        
        # Compute Tensor Components
        Tij = compute_time_domain_Tij(k_vec, r1, r2, d, 1.0) # rho_vac = 1.0
        
        # Compute Π value for this direction
        Pi_val = compute_time_domain_Π(Tij, xi, V)
        
        # Return weighted by differential solid angle element
        return Pi_val * sin(xi)
    end

    # Integrate from 0 to π
    val, err = quadgk(iso_integrand, 0.0, π)
    
    # Divide by 2 (result of ∫ 1/2 d(cosθ))
    return val / 2.0
end

