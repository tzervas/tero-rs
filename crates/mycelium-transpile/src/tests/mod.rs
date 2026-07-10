//! Test entry point (house rule: no inline tests in logic files — every `#[cfg(test)]` unit test
//! lives in this dedicated in-crate module, per CLAUDE.md "Test layout").

mod batch;
mod corpus;
mod diff;
mod emit;
mod invariant;
mod map;
mod prim_map;
mod reserved;
mod vet;
