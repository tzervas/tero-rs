//! In-crate white-box tests for `mycelium-workstack` (CLAUDE.md test layout: no tests in logic files).
//! Each submodule does `use crate::*` for white-box access; complex cases are data-driven tables, not
//! bespoke test-body logic.

mod arena;
mod budget;
mod guard_helper;
mod invariant;
mod isolation;
mod startup;
