"""
    I2_from_sincos(α::Float64, a::Float64, b::Float64, sa::Float64, ca::Float64, sb::Float64, cb::Float64) -> ComplexF64

Evaluate ∫_a^b τ² exp(iατ) dτ analytically using precomputed trigonometric values.
"""
@inline function I2_from_sincos(α::Float64, a::Float64, b::Float64, sa::Float64, ca::Float64, sb::Float64, cb::Float64)::ComplexF64
    iszero(a) && return I2_zero_lower_from_sincos(α, b, sb, cb) # Fast path
    iszero(α) && return ComplexF64((b^3 - a^3) / 3.0)

    # Condition must check the maximum dimensionless phase, not just α.
    max_x = abs(α) * max(abs(a), abs(b))

    if max_x < 1.0e-4
        # ∫ τ² (1 + iατ - α²τ²/2 - iα³τ³/6 + α⁴τ⁴/24 + iα⁵τ⁵/120) dτ
        a2, b2 = a*a, b*b
        a3, b3 = a2*a, b2*b
        a4, b4 = a2*a2, b2*b2
        a5, b5 = a4*a, b4*b
        a6, b6 = a5*a, b5*b
        a7, b7 = a6*a, b6*b
        a8, b8 = a7*a, b7*b

        return ComplexF64(
            (b3 - a3)/3.0 - α^2*(b5 - a5)/10.0 + α^4*(b7 - a7)/168.0,
            α*(b4 - a4)/4.0 - α^3*(b6 - a6)/36.0 + α^5*(b8 - a8)/960.0
        )
    end

    # General case: avoiding complex arithmetic
    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα

    a2, b2 = a*a, b*b

    # A(τ) = 2τ/α²
    # B(τ) = -τ²/α + 2/α³
    Aa = 2.0*a*invα2
    Ba = -a2*invα + 2.0*invα3

    Ab = 2.0*b*invα2
    Bb = -b2*invα + 2.0*invα3

    # F(τ) = (cos(ατ)A - sin(ατ)B) + i(sin(ατ)A + cos(ατ)B)
    return ComplexF64(
        (cb*Ab - sb*Bb) - (ca*Aa - sa*Ba),
        (sb*Ab + cb*Bb) - (sa*Aa + ca*Ba)
    )
end

"""
    I2(α::Float64, a::Float64, b::Float64) -> ComplexF64

Evaluate ∫_a^b τ² exp(iατ) dτ analytically.
"""
function I2(α::Float64, a::Float64, b::Float64)::ComplexF64
    iszero(α) && return ComplexF64((b^3 - a^3) / 3.0)
    sa, ca = sincos(α * a)
    sb, cb = sincos(α * b)
    return I2_from_sincos(α, a, b, sa, ca, sb, cb)
end

@inline function I2_zero_lower_from_sincos(α::Float64, b::Float64, s::Float64, c::Float64)::ComplexF64
    # ∫₀ᵇ τ² exp(i α τ) dτ

    b2 = b*b
    b3 = b2*b

    x = abs(α * b)

    # Small phase: Taylor expansion in ατ.
    if x < 1.0e-4
        b4 = b2*b2
        b5 = b4*b
        b6 = b5*b
        b7 = b6*b
        b8 = b7*b

        return ComplexF64(
            b3/3.0 - α^2*b5/10.0 + α^4*b7/168.0,
            α*b4/4.0 - α^3*b6/36.0 + α^5*b8/960.0
        )
    end

    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα

    # A(τ) = 2τ/α²
    # B(τ) = -τ²/α + 2/α³
    A = 2.0*b*invα2
    B = -b2*invα + 2.0*invα3

    # At lower limit τ=0, B(0) = 2/α³, contributing -i(2/α³) to the definite integral.
    return ComplexF64(
        c*A - s*B,
        s*A + c*B - 2.0*invα3
    )
end

"""
    I2_zero_lower(α::Float64, b::Float64) -> ComplexF64

Evaluate ∫₀ᵇ τ² exp(iατ) dτ analytically.
"""
@inline function I2_zero_lower(α::Float64, b::Float64)::ComplexF64
    s, c = sincos(α * b)
    return I2_zero_lower_from_sincos(α, b, s, c)
end
