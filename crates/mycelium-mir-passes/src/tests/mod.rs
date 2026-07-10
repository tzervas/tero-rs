//! In-crate tests for `mycelium-mir-passes` (M-797 test layout).
//!
//! One submodule per source module + a shared `common` builder module; white-box access via
//! `use crate::…`.

mod balance;
mod common;
mod corpus;
mod elision;
mod emit;
mod guards;
mod reuse;
