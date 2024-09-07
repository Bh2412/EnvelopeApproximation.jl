using EnvelopeApproximation.BubbleBasics
using FastSphericalHarmonics

# For a single bubble

function td_integrand(ϕ, μ:: SVector{2, Float64}, s:: Symbol):: Float64
    if s ≡ :trace
        return 1.
    elseif s ≡ :x
        return cos(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :y
        return sin(x[1]) * √(1 - x[2] ^ 2)
    elseif s ≡ :z
        return x[2]
    end
end

function td_integrand(x:: SVector{2, Float64}, td:: Tuple{Symbol, Symbol}):: Float64 
    td ≡ (:x, :x) && return cos(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:y, :y) && return sin(x[1]) ^ 2 * (1 - x[2] ^ 2)
    td ≡ (:z, :z) && return x[2] ^ 2
    ((td ≡ (:x, :y)) | (td ≡ (:y, :x))) && return cos(x[1]) * sin(x[1]) * (1 - x[2] ^ 2)
    return td_integrand(x, td[1]) * td_integrand(x, td[2])
end
