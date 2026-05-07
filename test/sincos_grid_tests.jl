using EnvelopeApproximation.RayCastingStressEnergyTensor
import EnvelopeApproximation.RayCastingStressEnergyTensor: compute_sincos_grid!
using Test

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
        @test S[1] ≈ sin(1.0 * 2.0 * 0.5) atol=1e-15
        @test C[1] ≈ cos(1.0 * 2.0 * 0.5) atol=1e-15
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
