#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

const EMPTY_ARC: (f64, f64) = (0., 0.);
const TWOPI: f64 = 2f64 * std::f64::consts::PI;
const FULL_CIRCLE: (f64, f64) = (0., TWOPI);
const EPS: f64 = 1e-8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeriodicInterval {
    pub psi: f64,
    pub delta: f64,
}

pub const EMPTY_ARC_INTERVAL: PeriodicInterval = PeriodicInterval { psi: 0.0, delta: 0.0 };

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Vector3> for (f64, f64, f64) {
    fn from(v: Vector3) -> (f64, f64, f64) {
        (v.x, v.y, v.z)
    }
}

impl From<(f64, f64)> for PeriodicInterval {
    fn from(t: (f64, f64)) -> PeriodicInterval {
        let (x, y) = t;
        PeriodicInterval {psi:x, delta: y}
    }
}

#[no_mangle]
pub extern "C" fn ring_dome_intersect(mu: f64, r: f64, nhat: Vector3, h: f64, dome_like: bool) -> PeriodicInterval {
    tuple_ring_dome_intersect(mu, r, nhat.into(), h, dome_like).into()
}

#[no_mangle]
pub extern "C" fn simple_intersection(p1: PeriodicInterval, p2: PeriodicInterval) -> PeriodicInterval {
    let start = p1.psi.max(p2.psi);
    let end = (p1.psi + p1.delta).min(p2.psi + p2.delta);
    let d = end - start;

    if d <= 0.0 {
        EMPTY_ARC_INTERVAL
    } else {
        PeriodicInterval { psi: start, delta: d }
    }
}


pub fn tuple_ring_dome_intersect(mu: f64, r: f64, nhat: (f64, f64, f64), h: f64, dome_like: bool) -> (f64, f64) {
    let (nx, ny, nz) = nhat;
    let murz = mu * r * nz;
    if (nx.abs() < EPS) & (ny.abs() < EPS) & ((nz.abs() - 1.0f64).abs() < EPS) {
        if h >= murz {
            if dome_like {
                return FULL_CIRCLE;
            } else {
                return EMPTY_ARC;
            }
        } else {
            if dome_like {
                return EMPTY_ARC;
            } else {
                return FULL_CIRCLE;
            }
        }
    }
    // Now nhat is not parallel to the sphere of the integration ring
    let s = (1f64 - mu.powi(2)).sqrt();
    let (d, sgn) = {
        let x = (h - murz) / (nx.powi(2) + ny.powi(2)).sqrt();
        (x.abs(), x.signum())
    };
    let dome_sign = if dome_like {-1.0f64} else {1.0f64};
    if d >= r * s {
        // Ring is either entirely in or entirely out, based on sign
        match sgn * dome_sign {
            1.0f64 => return EMPTY_ARC,
            -1.0f64 => return FULL_CIRCLE,
            _ => panic!("This should be unreachable, because we shouldn't have NaNs here!"),
        }
    }
    let alpha = ((ny * sgn).atan2(nx * sgn)).rem_euclid(TWOPI);
    let delta = (d / (r * s)).acos(); // The short arc
    match sgn * dome_sign {
        1.0f64 => return ((alpha - delta).rem_euclid(TWOPI), 2f64 * delta),
        -1.0f64 => return ((alpha + delta).rem_euclid(TWOPI), TWOPI - 2f64 * delta),
        _ => panic!("This should be unreachable, because we shouldn't have NaNs here!"),
    }
}

#[cfg(test)]
mod tests {

    #[derive(Debug, Copy, Clone)]
    struct Cos(f64);

    impl quickcheck::Arbitrary for Cos {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            Cos((f64::arbitrary(g).abs() % 1f64) * 2f64 - 1f64)
        }
    }

    const TWOPI: f64 = 2f64 * std::f64::consts::PI;

    #[quickcheck]
    fn analytic_solution(mf: Cos) -> quickcheck::TestResult {
        let mu = mf.0;
        if mu.is_nan() {
            return quickcheck::TestResult::discard();
        }
        let mumin = 1f64 / 8f64.sqrt() + 6f64.sqrt() / 4f64;
        let mupls = 1f64 / 8f64.sqrt() - 6f64.sqrt() / 4f64;
        let (_, b) = super::tuple_ring_dome_intersect(
            mu,
            1f64,
            (1f64 / 2f64.sqrt(), 0f64, 1f64 / 2f64.sqrt()),
            0.5f64,
            false,
        );
        let y = (0.5f64 + 2f64.sqrt() * mu - 2f64 * mu.powi(2)).sqrt();
        let x = 0.5f64.sqrt() - mu;
        let phipls = y.atan2(x);
        let phimin = (-y).atan2(x);
        let delta = phipls - phimin;
        let expect = {
            if mu <= mupls {
                0f64
            } else if mu >= mumin {
                TWOPI
            } else {
                delta
            }
        };
        println!("Expected {}", expect);
        println!("Actually got {}", b);
        quickcheck::TestResult::from_bool((b - expect).abs() < 1e-8)
    }
}
