module AngularIntegrationInterface

export AbstractAngularIntegrationPlan, integrate_angular

"""
Abstract type for strategies to integrate vector-valued functions 
over the unit sphere (4π steradians).
"""
abstract type AbstractAngularIntegrationPlan end

"""
    integrate_angular(plan::AbstractAngularIntegrationPlan, f_directional)

Computes ∫ f_directional(ϕ, θ) dΩ.
f_directional must accept (ϕ, θ) and return a Vector{ComplexF64}.
"""
function integrate_angular(plan::AbstractAngularIntegrationPlan, f_directional::Function)
    error("Not implemented for plan type $(typeof(plan))")
end

end