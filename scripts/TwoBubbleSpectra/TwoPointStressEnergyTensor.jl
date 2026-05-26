using StaticArrays
using LinearAlgebra

include(joinpath(@__DIR__, "AnalyticSolution.jl"))

# ------------------------------------------------------------
# 1. Cutoff
# ------------------------------------------------------------

struct GaussianLateTimeCutoff{T<:Real}
    tc::T
    t0::T
end

@inline function (C::GaussianLateTimeCutoff)(t::T) where {T<:Real}
    t <= C.tc && return one(T)
    return exp(-((t - C.tc)^2) / C.t0^2)
end

function two_bubble_cutoff(d::T; τ_over_d::T = T(1.2)) where {T<:Real}
    τ  = τ_over_d * d
    tc = T(0.9) * τ
    t0 = (τ - tc) / T(4)
    return τ, GaussianLateTimeCutoff(tc, t0)
end

# ------------------------------------------------------------
# 2. Time panels
# ------------------------------------------------------------
# Splits [0, τ] at the collision time (d/2, where the overlap geometry
# changes) and at tc (where the Gaussian cutoff starts decaying).
# Both are kinks in the integrand, so splitting here avoids wasted
# quadrature points straddling them.

function time_panels(d::T, τ::T, tc::T) where {T<:Real}
    pts = sort(unique(filter(x -> zero(T) <= x <= τ, T[zero(T), d / T(2), tc, τ])))
    return [(pts[i], pts[i+1]) for i in 1:length(pts)-1 if pts[i+1] > pts[i]]
end

# ------------------------------------------------------------
# 3. Time-integrated two-bubble amplitude
# ------------------------------------------------------------

function two_bubble_Tij_amplitude(
    ω::T,
    k̂::SVector{3,T},
    d::T;
    ρ_vac::T = one(T),
    τ_over_d::T  = T(1.2),
    include_time_phase::Bool = true,
    θ_scheme::IntegrationScheme = GaussLegendreScheme(64),
    t_scheme::IntegrationScheme = GaussLegendreScheme(96),
) where {T<:Real}

    τ, cutoff = two_bubble_cutoff(d; τ_over_d=τ_over_d)
    k_vec = ω .* k̂

    function integrand(t)
        Tij_t = compute_time_domain_Tij(k_vec, t, t, d, θ_scheme, ρ_vac)
        phase = include_time_phase ? cis(ω * t) : one(Complex{T})
        return cutoff(t) * phase * Tij_t
    end

    A = zero(SVector{6,Complex{T}})
    for (a, b) in time_panels(d, τ, cutoff.tc)
        A += integrate(t_scheme, integrand, a, b)
    end
    return A
end

# ------------------------------------------------------------
# 4. Direction average
# ------------------------------------------------------------

function direction_average_contraction(
    ω::T,
    d::T,
    projectors;
    ρ_vac::T     = one(T),
    τ_over_d::T  = T(1.2),
    include_time_phase::Bool = true,
    θ_scheme::IntegrationScheme = GaussLegendreScheme(128),
    t_scheme::IntegrationScheme = GaussLegendreScheme(32),
    μ_scheme::IntegrationScheme = GaussLegendreScheme(128),
) where {T<:Real}

    names = [name for (name, _) in projectors]
    Np    = length(projectors)

    function μ_integrand(μ)
        sinθ = sqrt(max(zero(T), one(T) - μ^2))
        k̂_μ  = SVector{3,T}(sinθ, zero(T), μ)

        Tij  = two_bubble_Tij_amplitude(
            ω, k̂_μ, d;
            ρ_vac=ρ_vac,
            τ_over_d=τ_over_d,
            include_time_phase=include_time_phase,
            θ_scheme=θ_scheme,
            t_scheme=t_scheme,
        )

        Tmat = Tij * Tij'
        vals = Vector{Complex{T}}(undef, Np)
        for (i, (_, Λ)) in enumerate(projectors)
            vals[i] = contract(Tmat, Λ(k̂_μ))
        end
        return vals
    end

    val = integrate(μ_scheme, μ_integrand, -one(T), one(T))
    return Dict(names[i] => T(2π) * val[i] for i in eachindex(names))
end
