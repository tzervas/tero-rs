//! In-crate white-box unit-test root for `std.io` logic modules (M-797 as-touched).
//!
//! One submodule per source module whose inline tests have been extracted. Declared in `lib.rs`
//! as `#[cfg(test)] #[path = "tests/mod.rs"] mod unit_tests;` (the crate already has its own
//! integration-level `mod tests` in `lib.rs`, so this module is named `unit_tests` to avoid a
//! name collision).

mod serialize;
