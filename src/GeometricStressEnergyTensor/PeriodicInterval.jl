export PeriodicInterval

struct PeriodicInterval
    ϕ1:: Float64
    Δ:: Float64
end

mod2π(ϕ:: Float64) = mod(ϕ, 2π)
∈(ϕ:: Float64, p:: PeriodicInterval):: Bool = mod2π(ϕ - p.ϕ1) <= p.Δ
approxempty(p:: PeriodicInterval):: Bool = p.Δ ≈ 0.
approxentire(p:: PeriodicInterval):: Bool = p.Δ ≈ 2π 

const EmptyArc:: PeriodicInterval = PeriodicInterval(0., 0.)
const FullCircle:: PeriodicInterval = PeriodicInterval(0., 2π)

function complement(p:: PeriodicInterval):: PeriodicInterval
    if approxempty(p)
        return FullCircle
    elseif approxentire(p)
        return EmptyArc
    else 
        return PeriodicInterval(mod2π(p.ϕ1 + p.Δ), 2π - p.Δ)
    end
end

function a(p:: PeriodicInterval)
    return p.ϕ1
end

function b(p:: PeriodicInterval)
    return mod2π(p.ϕ1 + p.Δ)
end
