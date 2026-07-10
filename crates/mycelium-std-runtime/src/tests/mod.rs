//! In-crate test modules for `mycelium-std-runtime` (M-797 test layout).
//!
//! One submodule per source module, each doing `use crate::…::*` for white-box access.
//! Logic files carry no test code — tests live here.
//!
//! `scheduler` has no test submodule here: the Scheduler itself (and its tests) relocated to
//! `mycelium-sched` (E25/M-862 dependency-cycle fix); `crate::scheduler` is now a thin re-export
//! (see `src/scheduler.rs`), so its behavior is exercised by `mycelium-sched`'s own test suite.
//!
//! `reclamation` and `supervision` likewise have no test submodules here: both modules (and their
//! tests) relocated to `mycelium-rt-abi` (M-883/M-884 runtime-ABI seam extraction); `crate::reclamation`
//! and `crate::supervision` are now thin re-exports (see `src/reclamation.rs` / `src/supervision.rs`),
//! so their behavior is exercised by `mycelium-rt-abi`'s own test suite.

pub mod composition;
pub mod dataflow;
pub mod guarantee_matrix;
pub mod network;
pub mod policy_mech;
pub mod r2_residual;
pub mod rc;
pub mod region;
pub mod scope_region;
