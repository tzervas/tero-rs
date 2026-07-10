//! In-crate test modules for `mycelium-sched` (M-797 test layout).
//!
//! One submodule per source module, each doing `use crate::…::*` for white-box access.
//! Logic files carry no test code — tests live here.

pub mod scheduler;
