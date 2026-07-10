//! In-crate white-box test modules (test layout rule: no tests in logic files; one submodule per
//! source module under test). Extracted as-touched from the logic files (M-797); these modules use
//! `use crate::<mod>::*` for white-box access to the logic module's `pub(crate)` items.

mod affine;
mod ambient;
mod ast;
mod checkty;
mod compiler_stage5_elab;
mod compiler_stage5_freevars;
mod compiler_stage5_mangle;
mod compiler_stage5_normpat;
mod compiler_stage5_register;
mod compiler_stage5_semcore;
mod compiler_stage5_typealg;
mod compiler_stage5_unify;
mod decision;
mod elab;
mod error;
mod eval;
mod fuse;
mod lexer;
mod lib_root;
mod marshal_support;
mod mono;
mod mono_tag;
mod nodule;
mod parse;
mod substrate;
mod totality;
mod usefulness;
mod via_ordering;
