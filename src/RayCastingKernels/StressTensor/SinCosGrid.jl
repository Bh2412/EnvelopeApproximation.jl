const DEFAULT_PHASE_RESET_INTERVAL = 64

"""
    compute_sincos_grid!(S, C, ks, c_val, τ_stop; reset_interval=64)

Populate

    S[q] = sin(ks[q] * c_val * τ_stop)
    C[q] = cos(ks[q] * c_val * τ_stop)

for linearly spaced `ks`, using a trigonometric recurrence with periodic
exact resets.
"""
function compute_sincos_grid!(
    S::AbstractVector{Float64},
    C::AbstractVector{Float64},
    ks::AbstractRange{<:Real},
    c_val::Float64,
    τ_stop::Float64;
    reset_interval::Int = DEFAULT_PHASE_RESET_INTERVAL,
)
    Nk = length(ks)

    @boundscheck begin
        length(S) == Nk || throw(DimensionMismatch("length(S) != length(ks)"))
        length(C) == Nk || throw(DimensionMismatch("length(C) != length(ks)"))
    end

    Nk == 0 && return nothing

    k0 = Float64(first(ks))
    Δk = Nk == 1 ? 0.0 : Float64((last(ks) - first(ks)) / (Nk - 1))

    multiplier = c_val * τ_stop

    sd, cd = sincos(Δk * multiplier)
    s, c = sincos(k0 * multiplier)

    @inbounds for q in 1:Nk
        S[q] = s
        C[q] = c

        q == Nk && break

        if reset_interval > 0 && q % reset_interval == 0
            k_next = Float64(ks[q + 1])
            s, c = sincos(k_next * multiplier)
        else
            s_new = s*cd + c*sd
            c_new = c*cd - s*sd
            s, c = s_new, c_new
        end
    end

    return nothing
end
