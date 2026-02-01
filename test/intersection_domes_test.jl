using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.EnvelopeAnalysis
import EnvelopeApproximation.EnvelopeAnalysis: unfold_periodic_bubbles, face_distance, edge_distance, 
    vertex_distance, wall_intersect_complement!, ∩
using Test
using StaticArrays
using LinearAlgebra

@testset "Intersection Domes Functions" begin
    @testset "Bubbles Intersection with Boxes" begin

        @testset "wall_intersect_complement!" begin
            domes = Vector{IntersectionDome}()
            R = 1.0
            
            # Scenario 1: Bubble completely INSIDE, no intersection
            # Wall at x=0, Center at x=2, Interior direction +1 (implies wall is at left)
            # Dist = (2 - 0)*1 = 2.0. Since 2.0 > R, no dome.
            wall_intersect_complement!(domes, 2.0, R, 0.0, 1.0, 1)
            @test isempty(domes)

            # Scenario 2: Bubble INSIDE, protruding out (Standard Intersection)
            # Wall at x=0, Center at x=0.5, Interior Dir +1
            # Dist = (0.5 - 0)*1 = 0.5. Since 0.5 < R, add dome.
            empty!(domes)
            wall_intersect_complement!(domes, 0.5, R, 0.0, 1.0, 1)
            @test length(domes) == 1
            d = domes[1]
            @test d.h ≈ 0.5
            @test d.dome_like == true
            # Wall is at 0, Center at 0.5. Vector from C to Wall is Negative.
            @test d.n̂ == Vec3(-1.0, 0.0, 0.0) 

            # Scenario 3: Bubble center OUTSIDE (Ghost Bubble Logic)
            # Wall at x=0, Center at x=-0.5, Interior Dir +1 (so center is "behind" wall)
            # Dist = (-0.5 - 0)*1 = -0.5. 
            # Logic flips: dist becomes 0.5, domelike becomes false.
            empty!(domes)
            wall_intersect_complement!(domes, -0.5, R, 0.0, 1.0, 1)
            @test length(domes) == 1
            d = domes[1]
            @test d.h ≈ 0.5
            @test d.dome_like == false
            # Wall is at 0, Center at -0.5. Vector from C to Wall is Positive.
            @test d.n̂ == Vec3(1.0, 0.0, 0.0)
        end

        @testset "∩(bubble, box_space) Integration" begin
            # Setup: Box size 10 (extends -5 to +5 on all axes)
            box = BoxSpace(10.0, Point3(0.0, 0.0, 0.0))
            R = 1.0

            # Case A: Bubble Safely Inside
            b_safe = Bubble(Point3(0.0, 0.0, 0.0), R)
            domes = ∩(b_safe, box)
            @test isempty(domes)

            # Case B: Protruding Right (+X)
            # Wall at +5. Center at 4.5.
            b_right = Bubble(Point3(4.5, 0.0, 0.0), R)
            domes = ∩(b_right, box)
            @test length(domes) == 1
            @test domes[1].h ≈ 0.5
            @test domes[1].dome_like == true
            @test domes[1].n̂ == Vec3(1.0, 0.0, 0.0) # Points +X towards wall

            # Case C: Protruding Left (-X)
            # Wall at -5. Center at -4.5.
            b_left = Bubble(Point3(-4.5, 0.0, 0.0), R)
            domes = ∩(b_left, box)
            @test length(domes) == 1
            @test domes[1].h ≈ 0.5
            @test domes[1].dome_like == true
            @test domes[1].n̂ == Vec3(-1.0, 0.0, 0.0) # Points -X towards wall

            # Case D: Corner Intersection (Right + Top + Front)
            # Walls at 5. Center at (4.5, 4.5, 4.5).
            b_corner = Bubble(Point3(4.5, 4.5, 4.5), R)
            domes = ∩(b_corner, box)
            @test length(domes) == 3
            # Verify all are caps (dome_like=true)
            @test all(d -> d.dome_like, domes)
            # Verify normals point +X, +Y, +Z
            normals = [d.n̂ for d in domes]
            @test Vec3(1.0, 0.0, 0.0) in normals
            @test Vec3(0.0, 1.0, 0.0) in normals
            @test Vec3(0.0, 0.0, 1.0) in normals

            # Case E: Ghost Bubble (Center Outside Box)
            # Wall at +5. Center at 5.5.
            # This represents a ghost entering from the right.
            b_ghost = Bubble(Point3(5.5, 0.0, 0.0), R)
            domes = ∩(b_ghost, box)
            @test length(domes) == 1
            d = domes[1]
            @test d.h ≈ 0.5
            @test d.dome_like == false # Should exclude the bulk
            # Normal should point from 5.5 back to 5.0 -> (-1, 0, 0)
            @test d.n̂ == Vec3(-1.0, 0.0, 0.0)
        end
    end

    @testset "Geometric Distance Functions" begin
        L = 10.0
        L_half = 5.0
        
        @testset "1. Face Distance (Unsigned)" begin
            # Right Face (+X)
            face_right = SVector(1, 0, 0)
            
            # Inside (x=4) -> Dist 1.0
            p_in = Point3(4.0, 0.0, 0.0)
            @test face_distance(p_in, face_right, L_half) ≈ 1.0
            
            # Outside (x=6) -> Dist -1.0
            p_out = Point3(6.0, 0.0, 0.0)
            @test face_distance(p_out, face_right, L_half) ≈ 1.0

            # Left Face (-X) [The Critical Test for the Bug Fix]
            face_left = SVector(-1, 0, 0)
            
            # Inside (x=-4) -> Dist to -5 is 1.0
            p_in_left = Point3(-4.0, 0.0, 0.0)
            @test face_distance(p_in_left, face_left, L_half) ≈ 1.0
            
            # Outside (x=-6) -> Dist to -5 is -1.0
            p_out_left = Point3(-6.0, 0.0, 0.0)
            @test face_distance(p_out_left, face_left, L_half) ≈ 1.0
        end

        @testset "2. Edge Distance (Euclidean)" begin
            # Edge at Top-Right (+X, +Y) -> Line at x=5, y=5
            edge = SVector(1, 1, 0)
            
            # Point on edge (5, 5, 0)
            p_on = Point3(5.0, 5.0, 0.0)
            @test edge_distance(p_on, edge, L_half) ≈ 0.0
            
            # Point inside (4, 4, 0) -> Dist to (5,5) is sqrt(1^2 + 1^2) = sqrt(2)
            p_in = Point3(4.0, 4.0, 0.0)
            @test edge_distance(p_in, edge, L_half) ≈ sqrt(2)
            
            # Z-invariance check (Infinite line)
            # Point (5, 5, 100) -> Should still be 0
            p_z = Point3(5.0, 5.0, 100.0)
            @test edge_distance(p_z, edge, L_half) ≈ 0.0

            # Edge at Bottom-Left (-X, -Y) -> Line at x=-5, y=-5
            edge_neg = SVector(-1, -1, 0)
            p_in_neg = Point3(-4.0, -4.0, 0.0)
            @test edge_distance(p_in_neg, edge_neg, L_half) ≈ sqrt(2)
        end

        @testset "3. Vertex Distance (Euclidean)" begin
            # Corner at (+X, +Y, +Z) -> (5, 5, 5)
            vert = SVector(1, 1, 1)
            
            # Point on vertex
            p_on = Point3(5.0, 5.0, 5.0)
            @test vertex_distance(p_on, vert, L_half) ≈ 0.0
            
            # Point inside (4, 4, 4) -> Dist to (5,5,5) is sqrt(1^2+1^2+1^2) = sqrt(3)
            p_in = Point3(4.0, 4.0, 4.0)
            @test vertex_distance(p_in, vert, L_half) ≈ sqrt(3)
            
            # Corner at (-X, -Y, -Z) -> (-5, -5, -5)
            vert_neg = SVector(-1, -1, -1)
            p_in_neg = Point3(-4.0, -4.0, -4.0)
            @test vertex_distance(p_in_neg, vert_neg, L_half) ≈ sqrt(3)
        end
    end

    # --- Mocking minimal structures for the test to run independently ---
    # (Skip this block if you are running inside your actual module)
    # include("IntersectionDome.jl") 
    # using .EnvelopeApproximation.BubbleBasics
    # using .EnvelopeApproximation.BubblesEvolution

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

    end
end;