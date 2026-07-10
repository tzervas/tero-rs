//! In-crate test modules for `mycelium-rt-abi` (M-797 test layout).
//!
//! One submodule per source module, each doing `use crate::…::*` for white-box access.
//! Logic files carry no test code — tests live here.
//!
//! Both submodules are relocated verbatim from `mycelium-std-runtime/src/tests/` (M-883/M-884 —
//! the runtime-ABI seam extraction); see that crate's `src/tests/mod.rs` for the relocation note.

pub mod reclamation;
pub mod supervision;
