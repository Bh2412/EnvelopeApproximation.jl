using Test
using QuadGK
using Bessels
using StaticArrays
using LinearAlgebra
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.StressEnergyTensorComponents
using EnvelopeApproximation.TwoPointStressEnergyTensorModule
using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
using EnvelopeApproximation.EnvelopeAnalysis: align_ẑ, symmetric_tensor_inverse_rotation

# --- Analytic reference ---

function compute_time_domain_Tij(k_vec::AbstractVector{T}, r1::T, r2::T, d::T, ρ_vac::T=1.0)::SVector{6, Complex{T}} where T<:Real
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
            j0 = besselj0(u)
            j1 = besselj1(u)
            j2 = besselj(2, u)
            return SVector{5, Complex{T}}(
                s^3 * (j0 - j2) * phase,
                s^3 * (j0 + j2) * phase,
                2 * s * c^2 * j0 * phase,
                -2im * s^2 * c * j1 * phase,
                s * (kz * c * j0 - im * kx * s * j1) * phase,
            )
        end
        res, _ = quadgk(integrand, theta_min, theta_max)
        return res
    end

    phase_shift1 = exp(im * kz * d / 2)
    phase_shift2 = exp(-im * kz * d / 2)
    I1 = integrate_bubble(r1, alpha1, π)
    I2 = integrate_bubble(r2, zero(T), π - alpha2)

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
                              ρ_vac::T=one(T), V::T=one(T))::SMatrix{6,6,Complex{T}} where T<:Real
    Tij = compute_time_domain_Tij(k_vec, r1, r2, d, ρ_vac)
    return (Tij * Tij') ./ V
end

# --- Numeric pipeline ---

function NumericTwoPointTij(k̂::AbstractVector{T}, ks::AbstractVector{<:Real},
                             r1::T, r2::T, d::T,
                             ρ_vac::T=one(T), V::T=one(T))::Array{ComplexF64, 3} where T<:Real
    R    = align_ẑ(Vec3(Float64.(k̂)))
    t    = max(Float64(r1), Float64(r2))
    nuc1 = (time = t - Float64(r1), site = Point3(0., 0., -Float64(d) / 2))
    nuc2 = (time = t - Float64(r2), site = Point3(0., 0.,  Float64(d) / 2))
    snapshot = R * BubblesSnapShot(Nucleation[nuc1, nuc2], t)
    space    = BoxSpace(cbrt(Float64(V)))
    plan     = VectorQuadGKPlan{7}()
    strategy = T_ij_AzimuthalReduction(plan, T_i_j_kernel(2))
    result_rotated = TwoPointStressEnergyTensor(t, t, collect(Float64, ks), snapshot, space,
                                                Vacuum(), strategy; ΔV = Float64(ρ_vac))
    D  = symmetric_tensor_inverse_rotation(R)
    Dt = D'
    return stack(D * result_rotated[:, :, ki] * Dt for ki in axes(result_rotated, 3))
end

# --- Tests ---

@testset "Analytic two-bubble solution (unequal sizes)" begin
    r1    = 5.0
    r2    = 3.0
    d     = 6.0
    ρ_vac = 1.0
    L     = 6 * (max(r1, r2) + d / 2)
    V     = L^3

    ks     = collect(LinRange(0.1, 2.0, 10))
    cos_ξs = [0., 1/3, 2/3, 1.]
    k̂s     = [normalize([sqrt(max(1 - cξ^2, 0.)), 0., cξ]) for cξ in cos_ξs]

    threshold = 1e-14
    reltol    = 1e-5

    for (cξ, k̂) in zip(cos_ξs, k̂s)
        @testset "cos(ξ) = $(round(cξ, digits=5))" begin
            numeric  = NumericTwoPointTij(k̂, ks, r1, r2, d, ρ_vac, V)
            analytic = [AnalyticTwoPointTij(k .* k̂, r1, r2, d, ρ_vac, V) for k in ks]

            for ki in eachindex(ks)
                N = numeric[:, :, ki]
                A = Matrix(analytic[ki])
                max_rel = maximum(
                    abs(A[i]) > threshold ? abs(N[i] - A[i]) / abs(A[i]) : 0.0
                    for i in CartesianIndices(A)
                )
                @test max_rel < reltol
            end
        end
    end
end