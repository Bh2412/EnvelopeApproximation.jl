# --- Analytic reference ---

using QuadGK
using FastGaussQuadrature
using StaticArrays
using Bessels
using LinearAlgebra

abstract type IntegrationScheme end

struct QuadGKScheme <: IntegrationScheme
    rtol::Float64
    atol::Float64
end

QuadGKScheme(; rtol=1e-8, atol=1e-10) = QuadGKScheme(rtol, atol)

function integrate(scheme::QuadGKScheme, integrand, a::Real, b::Real)
    val, _ = quadgk(integrand, a, b; rtol=scheme.rtol, atol=scheme.atol)
    return val
end

struct GaussLegendreScheme{T<:Real} <: IntegrationScheme
    nodes::Vector{T}
    weights::Vector{T}
end

function GaussLegendreScheme(N::Int; T=Float64)
    x, w = gausslegendre(N)
    return GaussLegendreScheme{T}(T.(x), T.(w))
end

function integrate(scheme::GaussLegendreScheme{T}, integrand, a::Real, b::Real) where {T<:Real}
    a_T, b_T = T(a), T(b)
    if b_T == a_T
        return zero(integrand(a_T))
    end

    mid  = (a_T + b_T) / 2
    half = (b_T - a_T) / 2

    x = scheme.nodes
    w = scheme.weights

    @inbounds begin
        y0  = integrand(mid + half * x[1])
        acc = (half * w[1]) * y0

        for q in 2:length(x)
            acc += (half * w[q]) * integrand(mid + half * x[q])
        end
    end

    return acc
end

function compute_time_domain_Tij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, scheme::IntegrationScheme, ρ_vac::T=1.0)::SVector{6, Complex{T}} where T<:Real
    kx = sqrt(k_vec[1]^2 + k_vec[2]^2)
    kz = k_vec[3]
    k2 = kx^2 + kz^2

    if d >= r1 + r2 || d <= abs(r1 - r2)
        alpha1 = zero(T)
        alpha2 = zero(T)
    else
        cos_a1 = clamp((r1^2 + d^2 - r2^2) / (2 * r1 * d), -one(T), one(T))
        cos_a2 = clamp((r2^2 + d^2 - r1^2) / (2 * r2 * d), -one(T), one(T))
        alpha1 = acos(cos_a1)
        alpha2 = acos(cos_a2)
    end

    function integrate_bubble(r, theta_min, theta_max)
        function integrand(theta)
            s, c = sincos(theta)
            u     = kx * r * s
            phase = exp(-im * kz * r * c)
            j0, j1, j2 = besselj(0:2, u)
            return SVector{5, Complex{T}}(
                s^3 * (j0 - j2) * phase,
                s^3 * (j0 + j2) * phase,
                2 * s * c^2 * j0 * phase,
                -2im * s^2 * c * j1 * phase,
                s * (kz * c * j0 - im * kx * s * j1) * phase,
            )
        end
        res = integrate(scheme, integrand, T(theta_min), T(theta_max))
        return res
    end

    phase_shift1 = exp(im * kz * d / 2)
    phase_shift2 = exp(-im * kz * d / 2)
    I1 = integrate_bubble(r1, alpha1, π)
    I2 = integrate_bubble(r2, zero(T), π- alpha2)

    factor1_kin = (π * ρ_vac / 3) * r1^3 * phase_shift1
    factor2_kin = (π * ρ_vac / 3) * r2^3 * phase_shift2
    factor1_pot = (r1 > 0 && k2 > 0) ? (2π * im * ρ_vac / k2) * r1^2 * phase_shift1 : zero(Complex{T})
    factor2_pot = (r2 > 0 && k2 > 0) ? (2π * im * ρ_vac / k2) * r2^2 * phase_shift2 : zero(Complex{T})

    Txx = factor1_kin * I1[1] + factor1_pot * I1[5] + factor2_kin * I2[1] + factor2_pot * I2[5]
    Tyy = factor1_kin * I1[2] + factor1_pot * I1[5] + factor2_kin * I2[2] + factor2_pot * I2[5]
    Tzz = factor1_kin * I1[3] + factor1_pot * I1[5] + factor2_kin * I2[3] + factor2_pot * I2[5]
    Txz = factor1_kin * I1[4] + factor2_kin * I2[4]

    return SVector{6, Complex{T}}(Txx, 0., Txz, Tyy, 0., Tzz)
end

function AnalyticTwoPointTij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T,
                              scheme::IntegrationScheme, ρ_vac::T=one(T), V::T=one(T))::SMatrix{6,6,Complex{T}} where T<:Real
    Tij = compute_time_domain_Tij(k_vec, r1, r2, d, scheme, ρ_vac)
    return (Tij * Tij') ./ V
end