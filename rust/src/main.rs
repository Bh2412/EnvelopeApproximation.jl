use benstuff::{tuple_ring_dome_intersect, PeriodicInterval, simple_intersection};
use std::time::Instant;

fn main() {
    let (a, b) = tuple_ring_dome_intersect(0.5f64, 1f64, (3f64, 0f64, 1f64), 4f64, false);
    println!("Result a was {:.32}", a);
    println!("Result b was {:.32}", b);

    let p1 = PeriodicInterval { psi: 0.1, delta: 0.4 }; // [0.1, 0.5)
    let p2 = PeriodicInterval { psi: 0.3, delta: 0.6 }; // [0.3, 0.9)

    let now = Instant::now();

    let iterations = 10_000_000;
    let mut result = PeriodicInterval { psi: 0.0, delta: 0.0 };

    for _ in 0..iterations {
        result = simple_intersection(p1, p2);
    }

    let elapsed = now.elapsed();
    println!("Result: {:?}", result);
    println!("Elapsed time for {} iterations: {:?}", iterations, elapsed);
    println!("Average time per call: {:?} ns", elapsed.as_nanos() / iterations as u128);

}
