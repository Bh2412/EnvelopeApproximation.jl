"""
    I3(α::Float64, a::Float64, b::Float64) -> ComplexF64

Evaluate ∫_a^b τ³ exp(iατ) dτ analytically.
"""
function I3_from_sincos(α::Float64, a::Float64, b::Float64, sa::Float64, ca::Float64, sb::Float64, cb::Float64)::ComplexF64
    iszero(a) && return I3_zero_lower_from_sincos(α, b, sb, cb) # Fast path
    iszero(α) && return ComplexF64((b^4 - a^4) / 4.0)

    # Condition must check the maximum dimensionless phase, not just α.
    max_x = abs(α) * max(abs(a), abs(b))

    if max_x < 1.0e-4
        # ∫ τ³ (1 + iατ - α²τ²/2 - iα³τ³/6 + α⁴τ⁴/24) dτ
        a2, b2 = a*a, b*b
        a3, b3 = a2*a, b2*b
        a4, b4 = a2*a2, b2*b2
        a5, b5 = a4*a, b4*b
        a6, b6 = a5*a, b5*b
        a7, b7 = a6*a, b6*b
        a8, b8 = a7*a, b7*b

        return ComplexF64(
            (b4 - a4)/4.0 - α^2*(b6 - a6)/12.0 + α^4*(b8 - a8)/192.0,
            α*(b5 - a5)/5.0 - α^3*(b7 - a7)/42.0
        )
    end

    # General case: avoiding complex arithmetic
    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα
    invα4 = invα2 * invα2

    a2, b2 = a*a, b*b
    a3, b3 = a2*a, b2*b

    # A(τ) = 3τ²/α² - 6/α⁴
    # B(τ) = -τ³/α + 6τ/α³
    Aa = 3.0*a2*invα2 - 6.0*invα4
    Ba = -a3*invα + 6.0*a*invα3

    Ab = 3.0*b2*invα2 - 6.0*invα4
    Bb = -b3*invα + 6.0*b*invα3

    # F(τ) = (cos(ατ)A - sin(ατ)B) + i(sin(ατ)A + cos(ατ)B)
    return ComplexF64(
        (cb*Ab - sb*Bb) - (ca*Aa - sa*Ba),
        (sb*Ab + cb*Bb) - (sa*Aa + ca*Ba)
    )
end

function I3(α::Float64, a::Float64, b::Float64)::ComplexF64
    sa, ca = sincos(α * a)
    sb, cb = sincos(α * b)
    return I3_from_sincos(α, a, b, sa, ca, sb, cb)
end

@inline function I3_zero_lower_from_sincos(α::Float64, b::Float64, s::Float64, c::Float64)::ComplexF64
    # ∫₀ᵇ τ³ exp(i α τ) dτ

    b2 = b*b
    b3 = b2*b
    b4 = b2*b2

    x = abs(α * b)

    # Small phase: Taylor expansion in ατ.
    # ∫ τ³ ∑ₘ (iατ)^m/m! dτ
    if x < 1.0e-4
        b5 = b4*b
        b6 = b5*b
        b7 = b6*b
        b8 = b7*b

        return ComplexF64(
            b4/4 - α*α*b6/12 + α^4*b8/192,
            α*b5/5 - α^3*b7/42
        )
    end

    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα
    invα4 = invα2 * invα2

    # Antiderivative:
    # e^{iαb}[b³/(iα) - 3b²/(iα)² + 6b/(iα)³ - 6/(iα)⁴] + 6/(iα)⁴
    #
    # Written as real arithmetic to avoid complex powers.
    A = 3.0*b2*invα2 - 6.0*invα4
    B = -b3*invα + 6.0*b*invα3

    return ComplexF64(
        c*A - s*B + 6.0*invα4,
        s*A + c*B
    )
end

function I3_zero_lower(α::Float64, b::Float64)::ComplexF64
    s, c = sincos(α * b)
    return I3_zero_lower_from_sincos(α, b, s, c)
end
