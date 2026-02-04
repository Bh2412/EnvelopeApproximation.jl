using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.Spaces
using EnvelopeApproximation.BoundaryConditions
using EnvelopeApproximation.EnvelopeAnalysis
import EnvelopeApproximation.EnvelopeAnalysis: unfold_periodic_bubbles, face_distance, edge_distance, 
    vertex_distance, wall_intersect_complement!, ∩, inanydome
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

    @testset "inanydome Logic (Normalized Inputs)" begin
        R = 1.0
        
        # Standard cap at the North Pole (+Z) 
        # Covering the region where projection > 0.5
        north_cap = IntersectionDome(0.5, Vec3(0.0, 0.0, 1.0), true)
        
        # Complement dome (Ghost/BallSpace logic)
        # n̂ points toward the boundary; h = 0.5, dome_like = false.
        # Region covered: projection < 0.5 (excludes points far from the boundary)
        ghost_comp = IntersectionDome(0.5, Vec3(0.0, 0.0, 1.0), false)

        @testset "Vector3 Overload" begin
            domes = [north_cap]
            
            # Test inside the cap (z = 1.0, so projection 1.0 > 0.5)
            v_top = normalize(Vec3(0.0, 0.0, 1.0))
            @test inanydome(v_top, R, domes) == true
            
            # Test just inside the boundary
            # For h=0.5, R=1.0, the critical z is 0.5
            v_inside = normalize(Vec3(0.0, 0.0, 0.6))
            @test inanydome(v_inside, R, domes) == true
            
            # Test outside the cap (Equator z=0.0 < 0.5)
            v_side = normalize(Vec3(1.0, 0.0, 0.0))
            @test inanydome(v_side, R, domes) == false
        end

        @testset "Spherical Coordinates (μ = cosθ)" begin
            domes = [north_cap]
            # Since n̂ is (0,0,1), the projection is μ * R.
            # μ=1.0 is the North Pole (Unit vector by definition)
            @test inanydome(1.0, 0.0, R, domes) == true   
            
            # μ=0.0 is the Equator (Unit vector by definition)
            @test inanydome(0.0, 0.0, R, domes) == false  
        end

        @testset "Complement (h > 0) Logic" begin
            # dome_like = false: covered if projection < h
            # With n̂ = +Z and h = 0.5, directions with z < 0.5 are "covered"
            domes = [ghost_comp]
            
            # Vector pointing away from the boundary (z = -1.0 < 0.5)
            v_back = Vec3(0.0, 0.0, -1.0)
            @test inanydome(v_back, R, domes) == true 
            
            # Side vector (z = 0.0 < 0.5)
            v_side = Vec3(0.0, 1.0, 0.0)
            @test inanydome(v_side, R, domes) == true
            
            # Vector pointing deep into the interior (z = 0.9 > 0.5)
            v_int = normalize(Vec3(0.0, 0.0, 0.9))
            @test inanydome(v_int, R, domes) == false 
        end
    end
end;