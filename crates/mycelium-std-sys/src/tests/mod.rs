//! In-crate test root for `mycelium-std-sys` (test-layout rule — `src/tests/`, one submodule per
//! source module being migrated). Each submodule does `use crate::…::*` for white-box access to
//! private items (e.g. the `parse_args` core mapping). Migrated lazily (M-797); only `sys` lives
//! here so far.

mod sys;
