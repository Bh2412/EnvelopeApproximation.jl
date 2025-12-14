using QuadGK
using Bessels

export compute_Pi, compute_Pi_single, compute_Pi_double

# --- Constants & Helper Functions ---

const Ε = 1e-8

"""
    safe_j0(x)
Implements j0(x) = sin(x)/x from Eq. (A23).
Limit x->0 is 1.
"""
function safe_j0(x, ε=Ε)
    if abs(x) < ε
        # Taylor: 1 - x^2/6 + x^4/120
        return 1.0 - x^2/6 + x^4/120
    end
    return sin(x) / x
end

"""
    safe_j1_div_x(x)
Implements j1(x)/x.
Paper Eq. (A23): j1(x) = (sin(x) - x*cos(x)) / x^2
So j1(x)/x = (sin(x) - x*cos(x)) / x^3
Limit x->0 is 1/3.
"""
function safe_j1_div_x(x, ε=Ε)
    if abs(x) < ε
        # Taylor: 1/3 - x^2/30 + x^4/840
        return 1.0/3.0 - x^2/30.0 + x^4/840.0
    end
    return (sin(x) - x*cos(x)) / x^3
end

"""
    safe_j2_div_x2(x)
Implements j2(x)/x^2.
Paper Eq. (A23): j2(x) = ((3-x^2)sin(x) - 3x*cos(x)) / x^3
So j2(x)/x^2 = ((3-x^2)sin(x) - 3x*cos(x)) / x^5
Limit x->0 is 1/15.
"""
function safe_j2_div_x2(x, ε=Ε)
    if abs(x) < ε
        # Taylor: 1/15 - x^2/210 + x^4/7560
        return 1.0/15.0 - x^2/210.0 + x^4/7560.0
    end
    return ((3 - x^2)*sin(x) - 3*x*cos(x)) / x^5
end

"""
    I_geom(td, r)
Geometric factor I(t_d, r). 
Uses Taylor expansion: 1 + (1+u^2)/4 * r + u^2/8 * r^2
"""
function I_geom(td, r, ε=1e-6)
    # 1. Handle absolute singularity (r=0, td=0)
    # This prevents the 0/0 NaN error
    if r < ε
        return 1.0 # Limit as r->0 is 1.0
    end

    # 2. Taylor expansion for small arguments
    if abs(td) ≤ ε && r ≤ ε
        u = td / r
        return 1.0 + (1.0 + u^2)/4.0 * r + (u^2)/8.0 * r^2
    end
    
    # 3. Exact formula
    return exp(td/2) + exp(-td/2) + (td^2 - (r^2 + 4r))/(4r) * exp(-r/2)
end

"""
    P_false_vacuum(T, td, r, Gamma_star)
Probability P(t_x, t_y, r) from Eq. (39), (41), (45).
Gamma(T) = Gamma_star * exp(T).
Assumes beta = 1 units.
"""
function P_false_vacuum(T, td, r, Gamma_star)
    # Gamma(T) in the paper is Gamma_* * e^T (assuming beta=1)
    Gamma_T = Gamma_star * exp(T)
    
    # Calculate I(x,y) from Eq. (45)
    I_val = 8 * pi * Gamma_T * I_geom(td, r)
    
    return exp(-I_val)
end

# --- Polynomial Functions F0, F1, F2 (Single Bubble) ---
# Eqs. (50)-(52)

function F0(r, td)
    return 2 * (r^2 - td^2)^2 * (r^2 + 6r + 12)
end

function F1(r, td)
    return 2 * (r^2 - td^2) * (
        -r^2 * (r^3 + 4r^2 + 12r + 24) +
        td^2 * (r^3 + 12r^2 + 60r + 120))
end

function F2(r, td)
    term1 = r^4 * (r^4 + 4r^3 + 20r^2 + 72r + 144)
    term2 = 2 * td^2 * r^2 * (r^4 + 12r^3 + 84r^2 + 360r + 720)
    term3 = td^4 * (r^4 + 20r^3 + 180r^2 + 840r + 1680)
    return 0.5 * (term1 - term2 + term3)
end

# --- Polynomial Function G (Double Bubble) ---
# Eq. (61)

function G_poly(td, r)
    return (r^2 - td^2) * ( (r^3 + 2r^2) + td * (r^2 + 6r + 12) )
end


# --- Integrands ---

function integrand_single(r, td, T, k, Gamma_star, kappa, rho0)
    # Based on Eq. (53)
    # Note: Pre-factors are applied outside, but Gamma(T) is part of the expression
    
    if r <= abs(td)
        return 0.0
    end

    prob = P_false_vacuum(T, td, r, Gamma_star)
    # Gamma(T) is handled in the prefactor in the calling function, 
    # but the paper includes it in the definition of Pi, so we need to be careful.
    # Looking at compute_Pi_single, we multiply by Gamma_T there.
    # So we do NOT multiply by Gamma_T here inside the integrand for the prefactor part,
    # BUT P_false_vacuum DOES depend on T.
    
    # Bessel terms
    j0 = safe_j0(k*r)
    j1_div_kr = safe_j1_div_x(k*r)
    j2_div_kr2 = safe_j2_div_x2(k*r)
    
    # F terms
    f0 = F0(r, td)
    f1 = F1(r, td)
    f2 = F2(r, td)
    
    bracket = j0 * f0 + j1_div_kr * f1 + j2_div_kr2 * f2
    
    # Assemble
    # Factor e^{-r/2} / r^3
    val = (exp(-r/2) / r^3) * prob * bracket
    
    return val
end

function integrand_double(r, td, T, k, Gamma_star, kappa, rho0)
    # Based on Eq. (62)
    
    if r <= abs(td)
        return 0.0
    end
    
    prob = P_false_vacuum(T, td, r, Gamma_star)
    
    j2_div_kr2 = safe_j2_div_x2(k*r)
    
    g_plus = G_poly(td, r)
    g_minus = G_poly(-td, r)
    
    # Assemble
    # Factor e^{-r} / r^4 * j2/(k^2 r^2)
    val = prob * (exp(-r) / r^4) * j2_div_kr2 * g_plus * g_minus
    
    return val
end

# --- Main Computation Functions ---

"""
    compute_Pi_single(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0)

Computes the single-bubble contribution to the unequal-time correlator Pi(tx, ty, k).
Based on Eq. (53) and velocity scaling Eq. (68).
"""
function compute_Pi_single(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0, kwargs...)
    # 1. Normalize inputs by beta
    # Time * beta -> dimensionless time
    # k / beta -> dimensionless wavenumber
    Tx = tx * beta
    Ty = ty * beta
    K_val = (k / beta) * v # Apply velocity scaling to k here: k -> vk
    
    td = Tx - Ty
    T_avg = (Tx + Ty) / 2
    
    # 2. Integration limits
    # The integration is over r from |td| to infinity (spacelike separation)
    r_min = abs(td)
    # Upper bound approximation (exp(-r/2) decays fast)
    r_max = r_min + 40.0 
    
    # 3. Perform Integration
    integral, err = quadgk(r -> integrand_single(r, td, T_avg, K_val, Gamma_star, kappa, rho0), r_min, r_max; kwargs...)
    
    # 4. Apply Prefactors and Velocity Scaling
    # Eq (53) prefactor: (4 pi^2 / 9) * kappa^2 * rho0^2 * Gamma(T)
    
    Gamma_T = Gamma_star * exp(T_avg)
    prefactor = (4 * pi^2 / 9) * kappa^2 * rho0^2 * Gamma_T
    
    # Velocity scaling Eq (68): v^3 * Pi(vk)
    result = v^3 * prefactor * integral
    
    return result
end

"""
    compute_Pi_double(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0)

Computes the double-bubble contribution to the unequal-time correlator Pi(tx, ty, k).
Based on Eq. (62) and velocity scaling Eq. (68).
"""
function compute_Pi_double(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0, kwargs...)
    Tx = tx * beta
    Ty = ty * beta
    K_val = (k / beta) * v # Velocity scaling k -> vk
    
    td = Tx - Ty
    T_avg = (Tx + Ty) / 2
    
    r_min = abs(td)
    r_max = r_min + 40.0 # exp(-r) decays faster than single bubble
    
    integral, err = quadgk(r -> integrand_double(r, td, T_avg, K_val, Gamma_star, kappa, rho0), r_min, r_max; kwargs...)
    
    # Eq (62) prefactor: (4 pi^3 / 9) * kappa^2 * rho0^2 * Gamma(T)^2
    Gamma_T = Gamma_star * exp(T_avg)
    prefactor = (4 * pi^3 / 9) * kappa^2 * rho0^2 * Gamma_T^2
    
    # Velocity scaling Eq (68): v^3 * Pi(vk)
    result = v^3 * prefactor * integral
    
    return result
end

"""
    compute_Pi(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0)

Computes the total unequal-time correlator Pi = Pi_single + Pi_double.
"""
function compute_Pi(tx, ty, k; v=1.0, beta=1.0, kappa=1.0, rho0=1.0, Gamma_star=1.0, kwargs...)
    s = compute_Pi_single(tx, ty, k; v=v, beta=beta, kappa=kappa, rho0=rho0, Gamma_star=Gamma_star, kwargs...)
    d = compute_Pi_double(tx, ty, k; v=v, beta=beta, kappa=kappa, rho0=rho0, Gamma_star=Gamma_star, kwargs...)
    return s + d
end
