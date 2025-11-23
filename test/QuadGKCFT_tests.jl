using Test
using EnvelopeApproximation.QuadGKCFT
using QuadGK
using StaticArrays
using LinearAlgebra

@testset "VectorQuadGKPlan Tests" begin

    @testset "Accuracy vs Scalar QuadGK" begin
        # 1. Define physics function (3 components)
        # Using SVector for best performance with the plan
        f(x) = SVector(exp(x), sin(x), cos(x))
        a, b = 1.0, 3.0
        
        # 2. Setup Plan for K=3
        plan = VectorQuadGKPlan{3}(rtol=1e-10, atol=1e-12)
        
        # 3. Define Wavenumbers
        ks = [0.0, 0.5, 1.0, 2.0, 10.0]
        
        # 4. Run Vectorized Transform
        # Result shape should be (length(ks), 3)
        result_matrix = fourier_modes(f, ks, plan, a, b)
        
        @test size(result_matrix) == (length(ks), 3)

        # 5. Verification Loop
        for (i_k, k) in enumerate(ks)
            # Verify each component (1, 2, 3) independently
            for i_comp in 1:3
                
                # Ground truth: Scalar QuadGK for single component * phase
                component_f(x) = f(x)[i_comp]
                
                # Note: fourier_modes computes ∫ f(x) * cis(-k*x) dx
                scalar_truth, _ = quadgk(x -> component_f(x) * cis(-k * x), a, b, rtol=1e-12)
                
                vector_val = result_matrix[i_k, i_comp]
                
                @test isapprox(vector_val, scalar_truth, rtol=1e-8)
            end
        end
    end
end