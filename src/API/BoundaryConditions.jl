module BoundaryConditions

export BoundaryCondition, Vacuum, Periodic

abstract type BoundaryCondition end

struct Vacuum <: BoundaryCondition end
struct Periodic <: BoundaryCondition end

end