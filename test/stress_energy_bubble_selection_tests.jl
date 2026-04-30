using Test
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.StressEnergyTensorComponents
using EnvelopeApproximation.QuadGKCFT
using EnvelopeApproximation.EnvelopeAnalysis: append_periodic_bubbles!, intersection_domes,
    original_bubble_groups
import EnvelopeApproximation.StressEnergyTensorComponents: bubble_∂iϕ∂jϕ_contribution!

function direct_∂iϕ∂jϕ_sum(ks, bubbles, space, ::Vacuum, plan, buffer, bubble_indices)
    V = zeros(ComplexF64, 6, length(ks))
    domes = intersection_domes(bubbles, space, Vacuum())
    for bubble_index in bubble_indices
        bubble_∂iϕ∂jϕ_contribution!(V, ks, bubbles[bubble_index], domes[bubble_index],
                                    plan, buffer)
    end
    return V
end

function direct_∂iϕ∂jϕ_sum(ks, bubbles, space, ::Periodic, plan, buffer, bubble_indices)
    V = zeros(ComplexF64, 6, length(ks))
    n_original = length(bubbles)
    origin_map = Int[]
    periodic_bubbles = append_periodic_bubbles!(collect(bubbles), space, origin_map)
    domes = intersection_domes(periodic_bubbles, space, Vacuum())
    groups = original_bubble_groups(origin_map, n_original)

    for original_index in bubble_indices
        for bubble_index in groups[original_index]
            bubble_∂iϕ∂jϕ_contribution!(V, ks, periodic_bubbles[bubble_index],
                                        domes[bubble_index], plan, buffer)
        end
    end
    return V
end

@testset "Stress-energy bubble selection" begin
    bubbles = [
        Bubble(Point3(-2.0, 0.0, 0.0), 0.5),
        Bubble(Point3( 2.0, 0.0, 0.0), 0.4),
    ]
    ks = [0.1, 0.2]
    space = BoxSpace(10.0)
    plan = VectorQuadGKPlan{6}()
    buffer = x̂_ix̂_j(10)

    for boundary_condition in (Vacuum(), Periodic())
        full = ∂iϕ∂jϕ(ks, bubbles, space, boundary_condition, plan, buffer)
        selected_sum =
            ∂iϕ∂jϕ(ks, bubbles, space, boundary_condition, plan, buffer; bubble_indices=1:1) +
            ∂iϕ∂jϕ(ks, bubbles, space, boundary_condition, plan, buffer; bubble_indices=2:2)

        @test full ≈ selected_sum
        @test ∂iϕ∂jϕ(ks, bubbles, space, boundary_condition, plan, buffer; bubble_indices=1:1) ≈
              direct_∂iϕ∂jϕ_sum(ks, bubbles, space, boundary_condition, plan, buffer, 1:1)
        @test ∂iϕ∂jϕ(ks, bubbles, space, boundary_condition, plan, buffer; bubble_indices=3:3) ==
              zeros(ComplexF64, 6, length(ks))
    end
end
