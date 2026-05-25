using JLD2
using Test

const REGRESSION_FIXTURE_DIR = joinpath(@__DIR__, "test_data", "regression")

regression_fixture_path(filename::AbstractString) =
    joinpath(REGRESSION_FIXTURE_DIR, filename)

function load_regression_fixture(filename::AbstractString)
    path = regression_fixture_path(filename)
    isfile(path) || error("Missing regression fixture: $path")
    return JLD2.load(path)
end

function test_regression_array(actual, fixture; key::AbstractString="expected",
                               rtol::Real=1.0e-12, atol::Real=1.0e-12)
    haskey(fixture, key) || error("Regression fixture is missing key '$key'. Available keys: $(collect(keys(fixture)))")
    
    expected = fixture[key]

    @test size(actual) == size(expected)
    @test eltype(actual) == eltype(expected)
    @test actual ≈ expected rtol=rtol atol=atol
    return nothing
end
