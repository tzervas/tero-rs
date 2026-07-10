//! **M-994 fix (b) cost spike (measurement, not a gate).** Reproduces the M-987 ~n^3 L1-eval cost
//! curve and lets it be re-measured before/after the `Arc`-structural-sharing change to
//! [`mycelium_l1::eval::L1Value`]'s `Data` clone.
//!
//! The shape mirrors the M-994 investigation: an O(n)-size accumulator (a `Cons` list) is referenced
//! across O(n) steps. `snoc` appends to the *end*, so a length-k list is rebuilt each step and its
//! tail variable `t` is `clone`d at every level of the recursion — the exact `eval_path` `v.clone()`
//! hot path. Before the fix, `L1Value::Data::clone` walks the whole spine (O(size)), so the whole
//! build is ~n^3; after the fix a `Data` clone is an `Arc` refcount bump (O(1)), so it drops a factor
//! of n. The driver `build` is tail-recursive (TCO'd since fix (a)), so the *driver* costs O(1)
//! host-stack depth — the measured cost is the `snoc`/clone work, isolated.
//!
//! Run (debug is fine — label it): `cargo test -p mycelium-l1 --test spike_m994_cost -- --ignored
//! --nocapture`. Sizes are overridable via `M994_SIZES=200,752,1252`. This is `#[ignore]`d: it is a
//! deliberate measurement (`Empirical`), not a correctness assertion, so it never runs in the normal
//! gate.

use std::time::Instant;

use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

/// Generate the cost program for a list of length `n`. `build` counts `n` down (tail-recursive →
/// TCO, O(1) depth) and `snoc`s a byte onto the growing accumulator each step; `snoc` recurses to
/// the list's end, cloning the tail variable at every level.
fn program(n: u32) -> String {
    format!(
        "nodule bench;\n\
         type L = Nil | Cons(Binary{{8}}, L);\n\
         fn snoc(xs: L, y: Binary{{8}}) => L = \
           match xs {{ Nil => Cons(y, Nil), Cons(h, t) => Cons(h, snoc(t, y)) }};\n\
         fn build(n: Binary{{32}}, acc: L) => L = \
           match eq(n, 0b{zero:032b}) {{ \
             0b1 => acc, \
             _ => build(sub_u(n, 0b{one:032b}), snoc(acc, 0b0000_0001)) \
           }};\n\
         fn main() => L = build(0b{n:032b}, Nil);",
        zero = 0u32,
        one = 1u32,
        n = n,
    )
}

/// Parse → check → monomorphize → L1-eval `main`, returning the wall-clock eval time.
fn time_eval(n: u32) -> std::time::Duration {
    let src = program(n);
    let nod = parse(&src).unwrap_or_else(|e| panic!("n={n}: parse failed: {e:?}"));
    let checked = check_nodule(&nod).unwrap_or_else(|e| panic!("n={n}: check failed: {e:?}"));
    let mono = monomorphize(&checked, "main")
        .unwrap_or_else(|e| panic!("n={n}: monomorphize failed: {e:?}"));
    // Raise the step budget well past the default 1M: `snoc`-append is ~O(n^2) *steps* (fuel is
    // charged per eval-step, independent of clone cost), so the large-n points need a bigger budget.
    // Fuel is charged identically before/after the clone change, so this does not perturb the
    // wall-clock comparison — it only lets the same n complete on both.
    let t0 = Instant::now();
    let _v = Evaluator::new(&mono)
        .with_fuel(50_000_000_000)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("n={n}: L1-eval failed: {e}"));
    t0.elapsed()
}

#[test]
#[ignore = "measurement spike (M-994 fix (b)); run explicitly with --ignored --nocapture"]
fn m994_cost_curve() {
    let sizes: Vec<u32> = std::env::var("M994_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("size"))
                .collect()
        })
        .unwrap_or_else(|| vec![200, 752, 1252]);

    println!("\n== M-994 fix (b) cost curve (debug build) ==");
    let mut pts: Vec<(u32, f64)> = Vec::new();
    for &n in &sizes {
        let d = time_eval(n);
        let secs = d.as_secs_f64();
        println!("  n = {n:>5} tokens : {secs:>10.4} s");
        pts.push((n, secs));
    }
    // Fit an exponent p from the two extreme points: t ~ n^p  ⇒  p = ln(t2/t1)/ln(n2/n1).
    if pts.len() >= 2 {
        let (n1, t1) = pts.first().copied().unwrap();
        let (n2, t2) = pts.last().copied().unwrap();
        if t1 > 0.0 && t2 > 0.0 {
            let p = (t2 / t1).ln() / (f64::from(n2) / f64::from(n1)).ln();
            println!("  fitted exponent p (t ~ n^p) over [{n1}, {n2}]: {p:.2}");
        }
    }
    println!("== end ==\n");
}
