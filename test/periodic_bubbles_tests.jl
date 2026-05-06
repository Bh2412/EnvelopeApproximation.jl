using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.EnvelopeAnalysis
import EnvelopeApproximation.EnvelopeAnalysis: unfold_periodic_bubbles, append_periodic_bubbles!, face_distance, edge_distance,
    vertex_distance, wall_intersect_complement!, ∩
using Test
using StaticArrays
using LinearAlgebra

@testset "Periodic Bubbles Generation Tests" begin

    # Define a standard box: Size 10, centered at 0
    # Extent is [-5, 5] along each axis
    L = 10.0
    box = BoxSpace(L, Point3(0., 0., 0.))
    
    # Radius = 1.0 (Small enough to test specific features without overlapping everything)
    R = 1.0

    @testset "1. Face Intersection (Right Wall)" begin
        # Bubble at x = 4.5. 
        # Right wall is at x = 5.0.
        # Distance to wall is 0.5. Since 0.5 < R, it overlaps.
        # It should NOT overlap any other wall (closest other wall is y=5 or z=5, dist=4.5)
        
        b = Bubble(Point3(4.5, 0.0, 0.0), R)
        
        ghosts = unfold_periodic_bubbles([b], box)
        
        # Expectation: 
        # 1. The original bubble is NOT in the list (unfold returns only ghosts? 
        #    Wait, check your implementation. My previous code returned ghosts ONLY 
        #    if calling `unfold_periodic_bubbles` inside the wrapper, or `periodic_copies!` 
        #    populates a list. Your `unfold` function iterates and calls `periodic_copies!`.
        #    Let's assume it returns the LIST OF GHOSTS as written in my last correction).
        
        @test length(ghosts) == 1
        
        ghost = ghosts[1]
        
        # Logic Check:
        # Bubble exits Right (+X). Ghost should enter from Left (-X).
        # Shift vector was (+L, 0, 0). 
        # New Center = Old Center - Shift = 4.5 - 10 = -5.5
        expected_center = Point3(-5.5, 0.0, 0.0)
        
        @test coordinates(ghost.center) ≈ coordinates(expected_center)
        @test ghost.radius == R
    end

    @testset "2. Edge Intersection (Top-Right Edge)" begin
        # Bubble at x = 4.5, y = 4.5, z = 0.0
        # Dist to Right Wall (x=5) = 0.5
        # Dist to Top Wall   (y=5) = 0.5
        # Dist to Edge       = sqrt(0.5^2 + 0.5^2) = 0.707
        # Since 0.707 < R (1.0), it touches the Edge.
        
        b = Bubble(Point3(4.5, 4.5, 0.0), R)
        
        ghosts = unfold_periodic_bubbles([b], box)
        
        # Expectation: 3 Ghosts
        # 1. Right Face Ghost (shifted x - L)
        # 2. Top Face Ghost (shifted y - L)
        # 3. Edge Ghost (shifted x - L, y - L)
        
        @test length(ghosts) == 3
        
        coords = [coordinates(g.center) for g in ghosts]
        
        # Expected locations:
        # (-5.5, 4.5, 0.0)  <- Face X ghost
        # (4.5, -5.5, 0.0)  <- Face Y ghost
        # (-5.5, -5.5, 0.0) <- Edge XY ghost
        
        expected_x_ghost = [-5.5, 4.5, 0.0]
        expected_y_ghost = [4.5, -5.5, 0.0]
        expected_xy_ghost = [-5.5, -5.5, 0.0]
        
        @test any(c -> c ≈ expected_x_ghost, coords)
        @test any(c -> c ≈ expected_y_ghost, coords)
        @test any(c -> c ≈ expected_xy_ghost, coords)
    end

    @testset "3. Vertex Intersection (Corner)" begin
        # Bubble at x = 4.5, y = 4.5, z = 4.5
        # Dist to each face = 0.5
        # Dist to Vertex = sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75) ≈ 0.866
        # Since 0.866 < R (1.0), it touches the Corner.
        
        b = Bubble(Point3(4.5, 4.5, 4.5), R)
        
        ghosts = unfold_periodic_bubbles([b], box)
        
        # Expectation: 7 Ghosts (2^3 - 1)
        # Faces: X, Y, Z
        # Edges: XY, XZ, YZ
        # Vertex: XYZ
        
        @test length(ghosts) == 7
        
        coords = [coordinates(g.center) for g in ghosts]
        
        # Check the Corner Ghost explicitly (the most complex one)
        # Should be shifted by (-L, -L, -L)
        expected_corner = [-5.5, -5.5, -5.5]
        
        @test any(c -> c ≈ expected_corner, coords)
    end

    @testset "4. Non-Intersection (Safe Zone)" begin
        # Bubble in the middle, far from walls
        b = Bubble(Point3(0.0, 0.0, 0.0), R)
        
        ghosts = unfold_periodic_bubbles([b], box)
        
        @test length(ghosts) == 0
    end
    
    @testset "5. Edge Miss (Face-Only Overlap)" begin
        # Bubble at x=4.5, y=4.5, but Radius is small (e.g., 0.6)
        # Face dists = 0.5. (Overlap faces)
        # Edge dist = 0.707. (Does NOT overlap edge, since 0.707 > 0.6)
        
        b = Bubble(Point3(4.5, 4.5, 0.0), 0.6)
        
        ghosts = unfold_periodic_bubbles([b], box)
        
        # Expectation:
        # Should spawn Face X ghost and Face Y ghost.
        # Should NOT spawn Edge ghost.
        
        @test length(ghosts) == 2
        
        coords = [coordinates(g.center) for g in ghosts]
        
        # Ghost 1: (-5.5, 4.5, 0)
        # Ghost 2: (4.5, -5.5, 0)
        # Edge ghost (-5.5, -5.5, 0) should NOT be present
        
        @test any(c -> c ≈ [-5.5, 4.5, 0.0], coords)
        @test any(c -> c ≈ [4.5, -5.5, 0.0], coords)
    end

    @testset "6. append_periodic_bubbles! preserves originals at their indices" begin
        # Two bubbles: one near a wall (will spawn ghosts), one safely interior.
        b1 = Bubble(Point3(4.5, 0.0, 0.0), R)   # near right wall → 1 ghost
        b2 = Bubble(Point3(0.0, 0.0, 0.0), R)   # interior → no ghosts

        bubbles = [b1, b2]
        origin_map = Int[]
        append_periodic_bubbles!(bubbles, box, origin_map)

        # Originals must still sit at indices 1 and 2.
        @test coordinates(bubbles[1].center) ≈ coordinates(b1.center)
        @test bubbles[1].radius == b1.radius
        @test coordinates(bubbles[2].center) ≈ coordinates(b2.center)
        @test bubbles[2].radius == b2.radius
    end

end