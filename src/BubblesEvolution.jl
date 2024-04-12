module BubblesEvolution

Nucleation = NamedTuple{(:time, :site)}

struct BubblesSnapshot
    nucleations:: Vector{Nucleation}
    t:: Float64
    radial_profile:: Function
end

function at_earlier_time(snap:: BubblesSnapshot, t:: Float64):: BubblesSnapshot
    nucleations = filter(nuc -> nuc[:time] <= t, snap.nucleations)
    return BubblesSnapshot(nucleations, t, snap.radial_profile)
end

function evolve(snap:: BubblesSnapshot, nucleations:: Vector{Nucleation}, Δt:: Float64)
    return BubblesSnapshot([snap.nucleations..., nucleations...], snap.t + Δt, snap.radial_profile)
end



end