//! In-crate white-box tests for `mycelium-stack` (CLAUDE.md test layout: no tests in logic files).
//! Each submodule does `use crate::*` for white-box access to the private `check_floor` decision;
//! complex cases are data-driven tables, not bespoke test-body logic.

mod deep_stack;
mod grow;
