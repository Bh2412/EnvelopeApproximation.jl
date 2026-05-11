using Test
using LinearAlgebra
using StaticArrays
using EnvelopeApproximation.StressEnergyTensorContractions

@testset "Stress-energy tensor contractions" begin
    k_x = SVector{3,Float64}(1.0, 0.0, 0.0)
    k_y = SVector{3,Float64}(0.0, 1.0, 0.0)
    k_z = SVector{3,Float64}(0.0, 0.0, 1.0)

    function make_tensor(;xx=0, xy=0, xz=0, yy=0, yz=0, zz=0)
        return ComplexF64[xx, xy, xz, yy, yz, zz]
    end

    projected_power(T::AbstractVector, k::AbstractVector) = contract(T * T', Λ_TT(k))

    tensor = SVector{6,ComplexF64}(1 + 2im, 3 - im, -2 + 4im, 5 + 0.5im, 7 - 3im, -11 + im)
    xx, xy, xz, yy, yz, zz = tensor

    @test Λ_TT(k_z) * tensor ≈ SVector{6,ComplexF64}(
        0.5 * (xx - yy),
        xy,
        0,
        0.5 * (yy - xx),
        0,
        0,
    )
    @test Λ_TT(k_y) * tensor ≈ SVector{6,ComplexF64}(
        0.5 * (xx - zz),
        0,
        xz,
        0,
        0,
        0.5 * (zz - xx),
    )
    @test Λ_TT(k_x) * tensor ≈ SVector{6,ComplexF64}(
        0,
        0,
        0,
        0.5 * (yy - zz),
        yz,
        0.5 * (zz - yy),
    )

    T = reshape(ComplexF64.(1:36), 6, 6)
    W_Λ = Matrix(contraction_weights) * Matrix(Λ_TT(k_z))
    @test contract(T, Λ_TT(k_z)) == tr(W_Λ * T)
    @test contract(T, Λ_TT, k_z) == tr(W_Λ * T)

    @test Λ_pp(k_z) == Λ_pp()
    @test Λ_pp()[1, 1] ≈ 1 / 9
    @test Λ_pp()[1, 4] ≈ 1 / 9
    @test Λ_pp()[1, 6] ≈ 1 / 9
    for i in (2, 3, 5), j in axes(Λ_pp(), 2)
        @test Λ_pp()[i, j] ≈ 0
        @test Λ_pp()[j, i] ≈ 0
    end

    @test Λ_T(k_z)[3, 3] ≈ 0.25
    @test Λ_T(k_z)[5, 5] ≈ 0.25
    @test Λ_T(k_z)[1, 1] ≈ 0.0

    @test Λ_σσ(k_z)[1, 1] ≈ 1 / 4
    @test Λ_σσ(k_z)[1, 4] ≈ 1 / 4
    @test Λ_σσ(k_z)[1, 6] ≈ -1 / 2
    @test Λ_σσ(k_z)[6, 6] ≈ 1

    @test Λ_pσ(k_z)[1, 1] ≈ -1 / 6
    @test Λ_pσ(k_z)[1, 4] ≈ -1 / 6
    @test Λ_pσ(k_z)[1, 6] ≈ -1 / 6
    @test Λ_pσ(k_z)[6, 1] ≈ 1 / 3

    @testset "Λ_TT quadratic projection" begin
        @testset "Positivity and Reality" begin
            for _ in 1:100
                T = rand(ComplexF64, 6)
                val = projected_power(T, k_z)

                @test isapprox(imag(val), 0.0, atol=1e-12)
                @test real(val) >= -1e-14
            end
        end

        @testset "Analytical Cases" begin
            T_trace = make_tensor(xx=1, yy=1, zz=1)
            @test projected_power(T_trace, k_z) ≈ 0.0 atol=1e-12

            T_plus = make_tensor(xx=1, yy=-1)
            @test projected_power(T_plus, k_z) ≈ 2.0 atol=1e-12

            T_cross = make_tensor(xy=1)
            @test projected_power(T_cross, k_z) ≈ 2.0 atol=1e-12

            T_mixed = make_tensor(xx=1, yy=-1, xy=1)
            @test projected_power(T_mixed, k_z) ≈ 4.0 atol=1e-12
        end

        @testset "Projection Property (Removal of non-GW terms)" begin
            T_base = make_tensor(xx=1, yy=-1)
            T_junk = make_tensor(xx=1, yy=-1, xz=5, yz=-3, zz=10)

            @test projected_power(T_junk, k_z) ≈ projected_power(T_base, k_z) atol=1e-12
        end

        @testset "Comparison with Direct Formula" begin
            for _ in 1:10
                T = rand(ComplexF64, 6)
                xx, xy, yy = T[1], T[2], T[4]

                expected = 0.5 * abs2(xx - yy) + 2.0 * abs2(xy)
                computed = projected_power(T, k_z)

                @test computed ≈ expected atol=1e-12
            end
        end
    end
end
