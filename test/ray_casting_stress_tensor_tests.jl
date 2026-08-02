"""
Unit tests for ray-casting T_ij computation.
"""

using EnvelopeApproximation
using EnvelopeApproximation.StressTensor
import EnvelopeApproximation.StressTensor: I3, I2, compute_sincos_grid!,
    prepare_source_modes!, KineticTerm, PotentialTerm, TotalStressTerm
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.TwoPointStressEnergyTensorModule: RayCastingCosineTimeIntegratedTwoPointStressTensor,
    RayCastingSphericalQuadrature,
    TimeIntegratedTwoPointStressEnergyTensor,
    TimeIntegratedTwoPointStressEnergyTensorPieces
using Test
using StaticArrays
using LinearAlgebra
using QuadGK
using StableRNGs
import EnvelopeApproximation.BubblesEvolution: sample_PT

include("regression_helpers.jl")

@testset "Ray Casting Stress Tensor" begin

    @testset "FourierStressTensorKernel constructor" begin
        kernel = FourierStressTensorKernel(KineticTerm(), CosineWeight(), [0.5, 1.0])

        @test kernel.mode_ws isa CosineModeWorkspace
        @test kernel.ΔV == 1.0
        @test kernel.v == 1.0

        weight = ComplexExponential([0, 0.5, -1.0])
        complex_kernel = FourierStressTensorKernel(KineticTerm(), weight, [0.5, 1.0])

        @test weight isa ComplexExponential{3}
        @test weight.ωs == [0.0, 0.5, -1.0]
        @test complex_kernel.mode_ws isa ComplexExponentialModeWorkspace{3}
        @test size(complex_kernel.mode_ws.phase) == (2, 3)

        source = (time=0.2, site=Point3(0.0, 0.0, 0.3))
        prepare_source_modes!(complex_kernel.mode_ws, weight, [0.5, 1.0], source)
        expected_phase = [cis(ω * source.time - k * source.site.coordinates[3])
                          for k in [0.5, 1.0], ω in weight.ωs]
        @test complex_kernel.mode_ws.phase ≈ expected_phase

        time_kernel = FourierStressTensorKernel(
            KineticTerm(), DiracDelta([0.1, 0.2]), [0.5, 1.0],
        )
        @test time_kernel isa TimeResolvedStressTensorKernel
        @test time_kernel.mode_ws isa DiracDeltaModeWorkspace
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: I₃ Analytic Integral
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "I3 Integral Evaluation" begin
        # α = 0 case
        @test isapprox(I3(0.0, 0.0, 1.0), (1.0^4 - 0.0^4) / 4.0, rtol=1e-10)
        @test isapprox(imag(I3(0.0, 0.0, 1.0)), 0.0, atol=1e-14)

        # Near-zero α uses Taylor series; should match α = 0 closely
        @test isapprox(I3(1e-12, 0.0, 1.0), I3(0.0, 0.0, 1.0), rtol=1e-8)

        # Swap limits changes sign
        @test isapprox(I3(1.5, 0.5, 1.5), -I3(1.5, 1.5, 0.5), rtol=1e-10)

        # Correctness against numerical quadrature — this catches wrong signs in the
        # antiderivative formula.
        for (α, a, b) in [(2.0, 0.0, 1.0), (0.5, 0.3, 2.0), (-1.0, 0.0, 3.0), (5.0, 1.0, 2.0)]
            ref, _ = quadgk(τ -> τ^3 * cis(α * τ), a, b; rtol=1e-12)
            @test isapprox(I3(α, a, b), ref, rtol=1e-8) broken=false
        end
    end

    @testset "I2 Integral Evaluation" begin
        # α = 0 case
        @test isapprox(I2(0.0, 0.0, 1.0), (1.0^3 - 0.0^3) / 3.0, rtol=1e-10)
        @test isapprox(imag(I2(0.0, 0.0, 1.0)), 0.0, atol=1e-14)

        # Near-zero α uses Taylor series; should match α = 0 closely
        @test isapprox(I2(1e-12, 0.0, 1.0), I2(0.0, 0.0, 1.0), rtol=1e-8)

        # Swap limits changes sign
        @test isapprox(I2(1.5, 0.5, 1.5), -I2(1.5, 1.5, 0.5), rtol=1e-10)

        # Correctness against numerical quadrature — this catches wrong signs in the
        # antiderivative formula.
        for (α, a, b) in [(2.0, 0.0, 1.0), (0.5, 0.3, 2.0), (-1.0, 0.0, 3.0), (5.0, 1.0, 2.0)]
            ref, _ = quadgk(τ -> τ^2 * cis(α * τ), a, b; rtol=1e-12)
            @test isapprox(I2(α, a, b), ref, rtol=1e-8) broken=false
        end
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: ray_T_ij returns a (A_plus, A_minus) tuple of correct shape
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "Single Bubble Ray-Casting" begin
        nuc = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc], 1.0)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()

        scheme = UniformSphericalCapScheme(4, 8)
        strategy = RayCastingSphericalQuadrature(scheme)

        ks = [0.5, 1.0, 2.0]
        acc = ray_T_ij(ks, snapshot, space, bc, KineticTerm(), CosineWeight(), scheme; ΔV=1.0)
        A_plus, A_minus = amplitudes(acc)

        @test size(A_plus)  == (6, 3)
        @test size(A_minus) == (6, 3)

        # A_plus and A_minus must differ (they carry opposite phase e^{+ikt} vs e^{-ikt})
        @test !isapprox(A_plus, A_minus, rtol=1e-6)

        # Single bubble with no collisions: A_minus = conj(A_plus) because
        # the integrand for A− is the complex conjugate of A+ (α → −α reflects as well).
        @test isapprox(A_minus, conj.(A_plus), rtol=1e-6)
    end

    @testset "L=10 TotalStressTerm regression through outer API" begin
        β = 1.0
        Γ_0 = 1.0e-4
        t_0 = 0.0
        tvp = 1 - 1.0e-5
        v_wall = 1.0
        egp = ExponentialGrowthProcess(β, Γ_0, tvp; t_0=t_0, v_wall=v_wall)

        L = 10.0 / β
        space = BoxSpace(L, Point3(0.0, 0.0, 0.0))
        bc = Periodic()
        snapshot = sample_PT(StableRNG(12312123), egp, space, bc; padding=0.0)

        ks = collect(range(2π / L; step=2π / L, length=3))
        strategy = RayCastingSphericalQuadrature(UniformSphericalCapScheme(6, 8))
        result = TimeIntegratedTwoPointStressEnergyTensor(
            ks, snapshot, space, bc, strategy, CosineWeight();
            term=TotalStressTerm(), ΔV=1.0, bubble_indices=1:3,
        )

        fixture = load_regression_fixture("ray_casting_total_stress_L10.jld2")

        @test fixture["api"] == "TimeIntegratedTwoPointStressEnergyTensor"
        @test fixture["term"] == "TotalStressTerm"
        @test fixture["weight"] == "CosineWeight"
        @test fixture["seed"] == 12312123
        @test fixture["L"] == L
        @test fixture["ks"] == ks
        @test length(snapshot.nucleations) == 40
        @test snapshot.t ≈ 8.429639302129399
        @test fixture["n_nucleations"] == length(snapshot.nucleations)
        @test fixture["snapshot_t"] ≈ snapshot.t
        test_regression_array(result, fixture; key="T", rtol=1.0e-12, atol=1.0e-12)
    end

    @testset "Generalized Ray-Casting API" begin
        nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        nuc2 = (time=0.2, site=Point3(1.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc1, nuc2], 2.0)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()

        scheme = UniformSphericalCapScheme(4, 8)
        ks = [0.5, 1.0]

        cosine_acc = ray_T_ij(
            ks, snapshot, space, bc,
            KineticTerm(), CosineWeight(), scheme; ΔV=1.0
        )

        @test cosine_acc isa CosineAccumulant

        kinetic_acc = ray_T_ij(
            ks, snapshot, space, bc,
            KineticTerm(), ConstantWeight(), scheme; ΔV=1.0
        )
        kinetic_cosine_acc = ray_T_ij(
            ks, snapshot, space, bc,
            KineticTerm(), CosineWeight(), scheme; ΔV=1.0
        )
        potential_acc = ray_T_ij(
            ks, snapshot, space, bc,
            PotentialTerm(), CosineWeight(), scheme; ΔV=1.0
        )
        constant_potential_acc = ray_T_ij(
            ks, snapshot, space, bc,
            PotentialTerm(), ConstantWeight(), scheme; ΔV=1.0
        )
        total_acc = ray_T_ij(
            ks, snapshot, space, bc,
            TotalStressTerm(), CosineWeight(), scheme; ΔV=1.0
        )
        constant_total_acc = ray_T_ij(
            ks, snapshot, space, bc,
            TotalStressTerm(), ConstantWeight(), scheme; ΔV=1.0
        )

        @test kinetic_acc isa ConstantAccumulant
        @test potential_acc isa CosineAccumulant
        @test constant_potential_acc isa ConstantAccumulant
        @test size(amplitudes(kinetic_acc)) == (6, 2)
        @test any(!iszero, amplitudes(kinetic_acc))
        @test amplitudes(total_acc)[1] ≈ amplitudes(kinetic_cosine_acc)[1] .+ amplitudes(potential_acc)[1]
        @test amplitudes(total_acc)[2] ≈ amplitudes(kinetic_cosine_acc)[2] .+ amplitudes(potential_acc)[2]
        @test amplitudes(constant_total_acc) ≈ amplitudes(kinetic_acc) .+ amplitudes(constant_potential_acc)
        @test_throws ArgumentError ray_T_ij(
            [0.0], snapshot, space, bc,
            PotentialTerm(), ConstantWeight(), scheme; ΔV=1.0
        )
        @test_throws ArgumentError ray_T_ij(
            [0.0], snapshot, space, bc,
            PotentialTerm(), CosineWeight(), scheme; ΔV=1.0
        )

        strategy = RayCastingSphericalQuadrature(scheme)
        for weight in (CosineWeight(), ConstantWeight())
            for term in (KineticTerm(), TotalStressTerm())
                result = TimeIntegratedTwoPointStressEnergyTensor(
                    ks, snapshot, space, bc, strategy, weight; term=term, ΔV=1.0
                )
                @test size(result) == (6, 6, 2)
                @test any(!iszero, result)
                for ki in 1:2
                    M = result[:, :, ki]
                    @test isapprox(M, M', rtol=1e-8)
                    for ij in 1:6
                        @test real(M[ij, ij]) >= -1e-12
                    end
                end
            end
        end
    end

    @testset "Complex-exponential temporal transform" begin
        nuc1 = (time=0.1, site=Point3(0.0, 0.0, 0.25))
        nuc2 = (time=0.3, site=Point3(1.0, 0.0, -0.4))
        snapshot = BubblesSnapShot([nuc1, nuc2], 1.5)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()
        scheme = UniformSphericalCapScheme(4, 8)

        ks = [0.5, 1.0]
        temporal_frequencies = [0.0; ks; -ks]
        weight = ComplexExponential(temporal_frequencies)

        for term in (KineticTerm(), PotentialTerm(), TotalStressTerm())
            A = amplitudes(ray_T_ij(
                ks, snapshot, space, bc, term, weight, scheme; ΔV=1.0,
            ))
            A_constant = amplitudes(ray_T_ij(
                ks, snapshot, space, bc, term, ConstantWeight(), scheme; ΔV=1.0,
            ))
            A_plus, A_minus = amplitudes(ray_T_ij(
                ks, snapshot, space, bc, term, CosineWeight(), scheme; ΔV=1.0,
            ))

            @test size(A) == (6, length(ks), length(temporal_frequencies))
            @test A[:, :, 1] ≈ A_constant
            for q in eachindex(ks)
                @test A[:, q, 1 + q] ≈ A_plus[:, q]
                @test A[:, q, 1 + length(ks) + q] ≈ A_minus[:, q]
            end
        end

        empty_snapshot = BubblesSnapShot(typeof(nuc1)[], 1.5)
        empty_A = amplitudes(ray_T_ij(
            ks, empty_snapshot, space, bc,
            KineticTerm(), weight, scheme; ΔV=1.0,
        ))
        @test size(empty_A) == (6, length(ks), length(temporal_frequencies))
        @test iszero(empty_A)
    end

    # ═════════════════════════════════════════════════════════════════════════════
    # Test: Two-point correlator is Hermitian and has positive diagonal
    # ═════════════════════════════════════════════════════════════════════════════

    @testset "RayCastingCosineTimeIntegratedTwoPointTensor" begin
        nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
        nuc2 = (time=0.2, site=Point3(1.0, 0.0, 0.0))
        snapshot = BubblesSnapShot([nuc1, nuc2], 2.0)
        space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
        bc = Periodic()

        scheme = UniformSphericalCapScheme(4, 8)
        strategy = RayCastingSphericalQuadrature(scheme)

        ks = [0.5, 1.0]
        result = RayCastingCosineTimeIntegratedTwoPointStressTensor(
            ks, snapshot, space, bc, strategy; ΔV=1.0
        )

        @test size(result) == (6, 6, 2)
        @test any(!iszero, result)

        for ki in 1:2
            M = result[:, :, ki]
            # Hermitian symmetry: M[i,j] = conj(M[j,i])
            @test isapprox(M, M', rtol=1e-8)
            # Diagonal entries are real non-negative (positive semi-definite diagonal)
            for ij in 1:6
                @test real(M[ij, ij]) >= -1e-12
            end
        end
    end

@testset "Named Terms API" begin
    nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
    nuc2 = (time=0.2, site=Point3(1.0, 0.0, 0.0))
    snapshot = BubblesSnapShot([nuc1, nuc2], 2.0)
    space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
    bc = Periodic()

    scheme = UniformSphericalCapScheme(4, 8)
    ks = range(0.5, 2.0, length=4)

    for weight in (CosineWeight(), ConstantWeight())
        acc_total = ray_T_ij(ks, snapshot, space, bc, TotalStressTerm(), weight, scheme; ΔV=1.0)
        acc_KV    = ray_T_ij(ks, snapshot, space, bc,
                             (:K => KineticTerm(), :V => PotentialTerm()),
                             weight, scheme; ΔV=1.0)

        @test acc_KV isa NamedTuple{(:K, :V)}
        @test acc_KV.K isa typeof(acc_total)
        @test acc_KV.V isa typeof(acc_total)

        if weight isa CosineWeight
            A_plus_total,  A_minus_total  = amplitudes(acc_total)
            A_plus_K,      A_minus_K      = amplitudes(acc_KV.K)
            A_plus_V,      A_minus_V      = amplitudes(acc_KV.V)
            @test A_plus_total  ≈ A_plus_K  .+ A_plus_V
            @test A_minus_total ≈ A_minus_K .+ A_minus_V
        else
            @test amplitudes(acc_total) ≈ amplitudes(acc_KV.K) .+ amplitudes(acc_KV.V)
        end
    end
end

@testset "Time-resolved ray-casting stress tensor" begin
    nuc = (time=0.0, site=Point3(0.0, 0.0, 0.0))
    snapshot = BubblesSnapShot([nuc], 1.0)
    space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
    bc = Periodic()
    scheme = UniformSphericalCapScheme(3, 6)
    ks = [0.5, 1.0]
    times = range(0.0, 1.0, length=2001)

    kinetic = ray_T_ij_at_times(
        times, ks, snapshot, space, bc;
        term=KineticTerm(), quadrature=scheme, ΔV=1.0,
    )
    potential = ray_T_ij_at_times(
        times, ks, snapshot, space, bc;
        term=PotentialTerm(), quadrature=scheme, ΔV=1.0,
    )
    total = ray_T_ij_at_times(
        times, ks, snapshot, space, bc;
        term=TotalStressTerm(), quadrature=scheme, ΔV=1.0,
    )

    @test size(amplitudes(total)) == (6, length(ks), length(times))
    @test amplitudes(total) ≈ amplitudes(kinetic) .+ amplitudes(potential)
    @test iszero(amplitudes(total)[:, :, 1])

    # The delta-time kernel integrates to the existing ConstantWeight kernel.
    dt = step(times)
    integrated = dt .* (
        dropdims(sum(amplitudes(total); dims=3); dims=3) .-
        0.5 .* amplitudes(total)[:, :, 1] .-
        0.5 .* amplitudes(total)[:, :, end]
    )
    reference = amplitudes(ray_T_ij(
        ks, snapshot, space, bc,
        TotalStressTerm(), ConstantWeight(), scheme; ΔV=1.0,
    ))
    @test integrated ≈ reference rtol=2.0e-6 atol=2.0e-8

    @test_throws ArgumentError ray_T_ij_at_times(
        [1.0, 0.0], ks, snapshot, space, bc; quadrature=scheme,
    )
    @test_throws ArgumentError ray_T_ij_at_times(
        times, [0.0], snapshot, space, bc;
        term=PotentialTerm(), quadrature=scheme,
    )
end

function total_piece(pieces::NamedTuple)
    first_outer = first(values(first(values(pieces))))
    total = zero(first_outer)

    for row in values(pieces)
        for arr in values(row)
            total .+= arr
        end
    end

    return total
end

@testset "TimeIntegratedTwoPointStressEnergyTensorPieces" begin
    nuc1 = (time=0.0, site=Point3(0.0, 0.0, 0.0))
    nuc2 = (time=0.2, site=Point3(1.0, 0.0, 0.0))
    snapshot = BubblesSnapShot([nuc1, nuc2], 2.0)
    space = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
    bc = Periodic()

    scheme = UniformSphericalCapScheme(4, 8)
    strategy = RayCastingSphericalQuadrature(scheme)
    ks = range(0.5, 2.0, length=4)

    for weight in (CosineWeight(), ConstantWeight())
        total = TimeIntegratedTwoPointStressEnergyTensor(
            ks, snapshot, space, bc, strategy, weight;
            term=TotalStressTerm(), ΔV=1.0
        )
        pieces = TimeIntegratedTwoPointStressEnergyTensorPieces(
            ks, snapshot, space, bc, strategy, weight; ΔV=1.0
        )

        @test pieces isa NamedTuple{(:K, :V)}
        @test pieces.K isa NamedTuple{(:K, :V)}

        # sum of all pieces equals the TotalStressTerm result
        @test total_piece(pieces) ≈ total

        # diagonal blocks are Hermitian (auto-correlations)
        for ki in 1:length(ks)
            @test isapprox(pieces.K.K[:, :, ki], pieces.K.K[:, :, ki]', rtol=1e-8)
            @test isapprox(pieces.V.V[:, :, ki], pieces.V.V[:, :, ki]', rtol=1e-8)
        end

        # off-diagonal blocks satisfy KV = VK†  (complex conjugate transpose)
        for ki in 1:length(ks)
            @test isapprox(pieces.K.V[:, :, ki], pieces.V.K[:, :, ki]', rtol=1e-8)
        end
    end
end

@testset "SinCos Grid" begin

    function reference_sincos(ks, c_val, τ_stop)
        multiplier = c_val * τ_stop
        S = [sin(k * multiplier) for k in ks]
        C = [cos(k * multiplier) for k in ks]
        return S, C
    end

    @testset "basic correctness" begin
        ks = range(0.1, 5.0, length=100)
        S = Vector{Float64}(undef, 100)
        C = Vector{Float64}(undef, 100)
        compute_sincos_grid!(S, C, ks, 1.5, 2.3)
        S_ref, C_ref = reference_sincos(ks, 1.5, 2.3)
        @test S ≈ S_ref atol=1e-12
        @test C ≈ C_ref atol=1e-12
    end

    @testset "single element" begin
        ks = range(1.0, 1.0, length=1)
        S = Vector{Float64}(undef, 1)
        C = Vector{Float64}(undef, 1)
        compute_sincos_grid!(S, C, ks, 2.0, 0.5)
        @test S[1] ≈ sin(1.0 * 2.0 * 0.5) atol=1e-13
        @test C[1] ≈ cos(1.0 * 2.0 * 0.5) atol=1e-13
    end

    @testset "zero start" begin
        ks = range(0.0, 3.0, length=50)
        S = Vector{Float64}(undef, 50)
        C = Vector{Float64}(undef, 50)
        compute_sincos_grid!(S, C, ks, 1.0, 1.0)
        S_ref, C_ref = reference_sincos(ks, 1.0, 1.0)
        @test S ≈ S_ref atol=1e-12
        @test C ≈ C_ref atol=1e-12
    end

    @testset "phase accumulation across reset boundary" begin
        # Use a length > reset_interval to exercise the reset path
        ks = range(0.5, 4.0, length=200)
        S = Vector{Float64}(undef, 200)
        C = Vector{Float64}(undef, 200)
        compute_sincos_grid!(S, C, ks, 0.8, 3.1)
        S_ref, C_ref = reference_sincos(ks, 0.8, 3.1)
        @test S ≈ S_ref atol=1e-10
        @test C ≈ C_ref atol=1e-10
    end

    @testset "custom reset_interval" begin
        ks = range(1.0, 10.0, length=300)
        S1 = Vector{Float64}(undef, 300)
        C1 = Vector{Float64}(undef, 300)
        S2 = Vector{Float64}(undef, 300)
        C2 = Vector{Float64}(undef, 300)
        compute_sincos_grid!(S1, C1, ks, 1.0, 1.0; reset_interval=16)
        compute_sincos_grid!(S2, C2, ks, 1.0, 1.0; reset_interval=128)
        S_ref, C_ref = reference_sincos(ks, 1.0, 1.0)
        @test S1 ≈ S_ref atol=1e-10
        @test C1 ≈ C_ref atol=1e-10
        @test S2 ≈ S_ref atol=1e-10
        @test C2 ≈ C_ref atol=1e-10
    end

    @testset "no reset (reset_interval=0)" begin
        ks = range(0.1, 2.0, length=100)
        S = Vector{Float64}(undef, 100)
        C = Vector{Float64}(undef, 100)
        compute_sincos_grid!(S, C, ks, 1.0, 1.0; reset_interval=0)
        S_ref, C_ref = reference_sincos(ks, 1.0, 1.0)
        # Pure recurrence accumulates drift — use a looser tolerance
        @test S ≈ S_ref atol=1e-10
        @test C ≈ C_ref atol=1e-10
    end

    @testset "dimension mismatch throws" begin
        ks = range(0.0, 1.0, length=10)
        S = Vector{Float64}(undef, 9)
        C = Vector{Float64}(undef, 10)
        @test_throws DimensionMismatch compute_sincos_grid!(S, C, ks, 1.0, 1.0)
        S2 = Vector{Float64}(undef, 10)
        C2 = Vector{Float64}(undef, 9)
        @test_throws DimensionMismatch compute_sincos_grid!(S2, C2, ks, 1.0, 1.0)
    end

    end

end  # @testset
;
