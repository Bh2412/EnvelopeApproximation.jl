.PHONY = clean

src/GeometricStressEnergyTensor/ring_dome_complement_intersection.so: 
	rustc rust/src/lib.rs --crate-type=cdylib -O -o src/GeometricStressEnergyTensor/ring_dome_complement_intersection.so
	strip src/GeometricStressEnergyTensor/ring_dome_complement_intersection.so
clean:
	rm src/GeometricStressEnergyTensor/ring_dome_complement_intersection.so

