using Base.Iterators

function exp_integral(t1:: Float64, t2:: Float64, β:: Float64)
    return (1 / β) * (exp(β * t2) - exp(β * t1))
end

function appropriate_gamma0(n0::Float64, β::Float64,
    Δt::Float64):: Float64
"""
The gamma0 value to produce the desired n0.
That is, this function implements a strategy to choose the gamma0
parameter of the exponential growth by selecting an appropriate
average amount of nucleations for the initial time window.
"""
    return β * n0 / (exp(β * Δt) - 1)
end

struct ExponentialGrowth <: NucleationLaw
    """
    generate time window limits equally spaced by the
    Δt.
    """
    β:: Float64
    Δt:: Float64
    Γ_0:: Float64
    t_0:: Float64

    function ExponentialGrowth(β:: Float64, Δt:: Float64;
                               Γ_0:: Union{Float64, Nothing} = nothing, 
                               t_0:: Float64 = 0.)
        if Γ_0 ≡ nothing
            Γ_0 = appropriate_gamma0(1., β, Δt)
        end
        return new(β, Δt, Γ_0, t_0)
    end
end

function Base.iterate(eg:: ExponentialGrowth)
    return ((eg.Δt, eg.Γ_0 * exp_integral(0., eg.Δt, eg.β)), eg.t_0 + eg.Δt)
end

function Base.iterate(eg:: ExponentialGrowth, t)
    return ((eg.Δt, eg.Γ_0 * exp_integral(t - eg.t_0, t + eg.Δt - eg.t_0, eg.β)), t + eg.Δt)
end

export ExponentialGrowth

