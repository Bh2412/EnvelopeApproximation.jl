module MonteCarloBigpi

using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.BubbleBasics: Bubble
using EnvelopeApproximation.EnvelopeAnalysis: IntersectionDome, intersection_domes, unfold_periodic_bubbles, append_periodic_bubbles!, original_bubble_groups, inanydome, n̂
using StaticArrays
using LinearAlgebra
using HCubature
using Random
using Measurements
using QuadGK
using StatsBase
using Bessels: sphericalbesselj

"""
    complex_to_real_covariance_optimized(mean_Z, Σ_H, Σ_P)

Efficiently converts complex statistics into real-valued formats for Measurements.jl.
Uses `reinterpret` to create the mean vector without allocation/copying.
"""
function complex_to_real_covariance_matrix(
    Σ_H::Matrix{ComplexF64},
    Σ_P::Matrix{ComplexF64}
)
    M = size(Σ_H, 1)
    S_plus = Σ_H + Σ_P
    S_minus = Σ_H - Σ_P

    cov_real = zeros(Float64, 2 * M, 2 * M)

    @inbounds for j in 1:M
        @inbounds for i in 1:j
            # Indices in the real matrix
            Xi, Yi = 2i - 1, 2i
            Xj, Yj = 2j - 1, 2j
            s_p = S_plus[i, j]
            s_m = S_minus[i, j]

            # 1. Cov(Xi, Xj) = 0.5 * Re(H + P)
            cov_real[Xi, Xj] = 0.5 * real(s_p)

            # 2. Cov(Yi, Yj) = 0.5 * Re(H - P)
            cov_real[Yi, Yj] = 0.5 * real(s_m)

            # 3. Cov(Xi, Yj) = 0.5 * Im(P - H)
            cov_real[Xi, Yj] = -0.5 * imag(s_m)

            # 4. Cov(Yi, Xj) = 0.5 * Im(P + H)
            # For the i == j case, this write to the lower triangle. When calling "Symmetric" later on we "run-over" this value.
            cov_real[Yi, Xj] = 0.5 * imag(S_plus[i, j])
        end
    end
    return Symmetric(cov_real, :U)
end

function decompose_covariance(covariance_matrix:: AbstractMatrix{Float64}):: Tuple{Vector{Float64}, Symmetric{Float64, Matrix{Float64}}}
    σ = sqrt.(diag(covariance_matrix))
    # Treating carefully the σ = 0 case
    inv_σ = map(x -> x > 1e-20 ? 1.0 / x : 0.0, σ)
    correlation_matrix = covariance_matrix .* (inv_σ * inv_σ')
    for i in eachindex(σ)
        if σ[i] <= 1e-20
            correlation_matrix[i, i] = 1.0
        end
    end
    return (σ, Symmetric(correlation_matrix, :U))
end

const EPS:: Float64 = 1e-10

function nullify_negative_eigenvalues(A::AbstractMatrix{Float64}; tol::Float64=EPS)::Symmetric{Float64, Matrix{Float64}}
    # This guarantees eigenvalues are Real, avoiding complex type errors.
    F = eigen(Symmetric(A))
    max_eigenvalue = maximum(abs, F.values)
    safe_floor = max(tol, max_eigenvalue * 1e-14)

    clean_vals = map(F.values) do v
        if v < -tol
            throw(ArgumentError("Matrix is not PSD. Found significant negative eigenvalue: $v"))
        else
            return max(v, safe_floor)
        end
    end
    
    # Spectral Reconstruction
    return Symmetric(F.vectors * Diagonal(clean_vals) * F.vectors')
end

function correlated_complex_measurements(
    mean_Z::Vector{ComplexF64},
    Σ_H::Matrix{ComplexF64},
    Σ_P::Matrix{ComplexF64}
)
    M = length(mean_Z)

    # Convert mean to real-valued without allocation
    mean_real = reinterpret(Float64, mean_Z)

    # Convert covariance matrices
    σ, correlation_matrix = complex_to_real_covariance_matrix(Σ_H, Σ_P) |> decompose_covariance
    correlation_matrix = nullify_negative_eigenvalues(correlation_matrix)

    # Create Measurements
    measurements_real = Measurements.correlated_values(mean_real, σ, correlation_matrix)
    # Reinterpret back to complex Measurements
    measurements_complex = similar(mean_Z, Complex{Measurement{Float64}})

    @inbounds for i in 1:M
        re = measurements_real[2i-1]
        im = measurements_real[2i]
        measurements_complex[i] = complex(re, im)
    end

    return measurements_complex
end

function first_5_spherical_bessels(z:: Real):: NTuple{5, Float64}
    j₀ = sphericalbesselj(0, z)
    j₁ = sphericalbesselj(1, z)
    invz = 1 / z
    j2 = 3 * j₁ * invz - j₀
    j3 = (5 * j2) * invz - j₁
    j4 = (7 * j3) * invz - j2
    return (j₀, j₁, j2, j3, j4)
end

function small_z_coeffs(z::Real)
    z2 = z * z
    z4 = z2 * z2
    
    c1 = π * (-8//15 + 4//21 * z2 - 11//945 * z4)
    c2 = π * (8//5 - 44//105 * z2 + 23//945 * z4)
    c3 = π * (16//35 * z2 - 32//945 * z4)
    c4 = π * (-16//105 * z2 + 2//189 * z4)
    c5 = π * (2//945 * z4)
    
    return SVector{5, Float64}(c1, c2, c3, c4, c5)
end

function coeffs(z:: Real):: SVector{5,Float64}
    if z < 1e-4
        return small_z_coeffs(z)
    end

    j₀, j₁, j₂, j₃, j₄ = first_5_spherical_bessels(z)
    return SVector{5,Float64}(4π * ((- 1. / 2) * j₀ + j₁ / z + j₂ / (2 * (z ^ 2))), 
                              4π * (j₀ -2 * j₁ / z + j₂ / (z ^ 2)),
                              8π * (j₂ - j₃ / z),
                              4π * (-1. / 2 * j₂ - 1. / 2 * j₃ / z), 
                              2π * j₄)
end

function dot_products(x̂₁:: SVector{3,Float64}, x̂₂:: SVector{3,Float64}, n̂:: SVector{3,Float64}):: SVector{5,Float64}
    c₁ = dot(x̂₁, n̂)
    c₂ = dot(x̂₂, n̂)
    c_12 = dot(x̂₁, x̂₂)
    return SVector{5,Float64}(1., c_12 ^ 2, c_12 * c₁ * c₂, c₁ ^ 2 + c₂ ^ 2, c₁ ^ 2 * c₂ ^ 2)
end

function integrated_projected(x̂₁:: SVector{3, Float64}, x̂₂:: SVector{3, Float64}, n̂:: SVector{3, Float64}, z:: Float64):: Float64
    return dot(coeffs(z), dot_products(x̂₁, x̂₂, n̂))
end

function integrated_projected(x̂₁:: SVector{3, Float64}, x̂₂:: SVector{3, Float64}, n̂:: SVector{3, Float64}, z:: AbstractVector{Float64})
    prods =  dot_products(x̂₁, x̂₂, n̂)
    v = similar(z)
    @inbounds for i in eachindex(z)
        v[i] = dot(coeffs(z[i]), prods)
    end
    return v
end

function Π(t1:: Float64, t2:: Float64, ks:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, space::BoxSpace, boundary_condition:: Periodic;
    N_samples:: Int=1_000_000, rng:: AbstractRNG=Random.default_rng(), ΔV:: Float64=1.0):: Vector{Measurement}
    
    bubbles1 = append_periodic_bubbles!(collect(current_bubbles(snapshot, t1)), space)
    bubbles2 = append_periodic_bubbles!(collect(current_bubbles(snapshot, t2)), space)

    if isempty(bubbles1) | isempty(bubbles2)
        return zeros(Measurement{Float64}, length(ks))
    end

    # Since we include the periodic copies, going forward we may treat the problem as if it 
    # is with Vacuum boundary conditions
    domes_dict1 = intersection_domes(bubbles1, space, Vacuum())
    domes_dict2 = intersection_domes(bubbles2, space, Vacuum())

    # 1. Setup Bubble Selection Probabilities (Proportional to R^3)
    weights1 = map(b -> b.radius^3, bubbles1)
    weights2 = map(b -> b.radius^3, bubbles2)
    prefactor = begin
        total_weight1 = sum(weights1)
        total_weight2 = sum(weights2)
        4π / 9 * (ΔV ^ 2) * total_weight1 * total_weight2 / EnvelopeApproximation.BubblesEvolution.volume(space)
    end # This prefactor transforms the result from an average to an integral over the domain

    # Create a sampler for efficient selection
    # (Aliasing or simple categorical sampling)
    bubble_indices1 = 1:length(bubbles1)
    bubble_sampler1 = Weights(weights1) # StatsBase handles the normalization

    bubble_indices2 = 1:length(bubbles2)
    bubble_sampler2 = Weights(weights2) # StatsBase handles the normalization

    N = length(ks)
    val = zeros(Float64, N) # Temporary storage for each sample's result
    S_1 = zeros(Float64, N) # Nominal sum
    S_S = zeros(Float64, N, N) # Z * Zᵀ

    # Monte Carlo Loop
    for _ in 1:N_samples
        idx1 = sample(rng, bubble_indices1, bubble_sampler1)
        bubble1 = bubbles1[idx1]
        domes1 = domes_dict1[idx1]

        idx2 = sample(rng, bubble_indices2, bubble_sampler2)
        bubble2 = bubbles2[idx2]
        domes2 = domes_dict2[idx2]

        # Sample two points on the sphere
        μ₁ = 2.0 * rand(rng) - 1.0
        ϕ₁ = 2π * rand(rng)
        x̂₁ = n̂(μ₁, ϕ₁)

        μ₂ = 2.0 * rand(rng) - 1.0
        ϕ₂ = 2π * rand(rng)
        x̂₂ = n̂(μ₂, ϕ₂)

        # Check geometric overlap
        if inanydome(x̂₁, bubble1, domes1) || inanydome(x̂₂, bubble2, domes2)
            continue # Contribution is 0
        end

        # Evaluate integrand for each k
        r = bubble1.radius * x̂₁ + coordinates(bubble1.center) - (bubble2.radius * x̂₂ + coordinates(bubble2.center))
        d = norm(r)
        _n̂ = r / d
        z = ks .* d
        val = integrated_projected(x̂₁, x̂₂, _n̂, z) * prefactor
        S_1 .+= val
        BLAS.syr!('U', 1.0, val, S_S)
    end
    # Final Statistics
    mean_Π = S_1 ./ N_samples
    Σ_Π = Symmetric((S_S - (1 / N_samples) * (S_1 * S_1')) ./ (N_samples * (N_samples - 1))) # Covariance matrix
    σ, corr = decompose_covariance(Σ_Π)
    corr = nullify_negative_eigenvalues(corr)
    return Measurements.correlated_values(mean_Π, σ, corr)
end

function Π_single(t1::Float64, t2::Float64, ks::AbstractVector{Float64}, snapshot::BubblesSnapShot, space::BoxSpace, ::Periodic;
    N_samples::Int=1_000_000, rng::AbstractRNG=Random.default_rng(), ΔV::Float64=1.0)::Vector{Measurement}

    # Bubbles shared between t1 and t2 are those present at the earlier time.
    t_early = min(t1, t2)
    n_shared = length(current_bubbles(snapshot, t_early))

    if n_shared == 0
        return zeros(Measurement{Float64}, length(ks))
    end

    origin_map1 = Int[]
    bubbles1 = append_periodic_bubbles!(collect(current_bubbles(snapshot, t1)), space, origin_map1)
    origin_map2 = Int[]
    bubbles2 = append_periodic_bubbles!(collect(current_bubbles(snapshot, t2)), space, origin_map2)

    domes_dict1 = intersection_domes(bubbles1, space, Vacuum())
    domes_dict2 = intersection_domes(bubbles2, space, Vacuum())

    copy_groups1 = original_bubble_groups(origin_map1, n_shared)
    copy_groups2 = original_bubble_groups(origin_map2, n_shared)

    # Weight for original bubble i: R_i(t1)^3 * R_i(t2)^3 * n_i1 * n_i2.
    # The n_i1 * n_i2 factor absorbs the uniform copy sub-sampling, so no per-sample correction is needed.
    weights = [bubbles1[i].radius^3 * bubbles2[i].radius^3 *
               length(copy_groups1[i]) * length(copy_groups2[i]) for i in 1:n_shared]
    W = sum(weights)
    prefactor = 4π / 9 * ΔV^2 * W / EnvelopeApproximation.BubblesEvolution.volume(space)

    orig_sampler = Weights(weights)

    N = length(ks)
    val = zeros(Float64, N)
    S_1 = zeros(Float64, N)
    S_S = zeros(Float64, N, N)

    for _ in 1:N_samples
        i = sample(rng, 1:n_shared, orig_sampler)

        copies1_i = copy_groups1[i]
        copies2_i = copy_groups2[i]
        idx1 = copies1_i[rand(rng, 1:length(copies1_i))]
        idx2 = copies2_i[rand(rng, 1:length(copies2_i))]

        bubble1 = bubbles1[idx1]
        domes1  = domes_dict1[idx1]
        bubble2 = bubbles2[idx2]
        domes2  = domes_dict2[idx2]

        μ₁ = 2.0 * rand(rng) - 1.0
        ϕ₁ = 2π * rand(rng)
        x̂₁ = n̂(μ₁, ϕ₁)

        μ₂ = 2.0 * rand(rng) - 1.0
        ϕ₂ = 2π * rand(rng)
        x̂₂ = n̂(μ₂, ϕ₂)

        if inanydome(x̂₁, bubble1, domes1) || inanydome(x̂₂, bubble2, domes2)
            continue
        end

        r = bubble1.radius * x̂₁ + coordinates(bubble1.center) - (bubble2.radius * x̂₂ + coordinates(bubble2.center))
        d = norm(r)
        _n̂ = r / d
        z = ks .* d
        val = integrated_projected(x̂₁, x̂₂, _n̂, z) * prefactor
        S_1 .+= val
        BLAS.syr!('U', 1.0, val, S_S)
    end

    mean_Π = S_1 ./ N_samples
    Σ_Π = Symmetric((S_S - (1 / N_samples) * (S_1 * S_1')) ./ (N_samples * (N_samples - 1)))
    σ, corr = decompose_covariance(Σ_Π)
    corr = nullify_negative_eigenvalues(corr)
    return Measurements.correlated_values(mean_Π, σ, corr)
end

end