//! In-crate white-box tests for `mycelium-proj` (test-layout rule: no tests in logic files). One
//! submodule per source module, each `use crate::<module>::*` for private-item access. Extracted
//! from the formerly-inline `#[cfg(test)] mod tests` blocks as part of M-790 (the as-touched
//! retrofit — M-797).

mod cert_scope;
mod header;
mod manifest;
mod resolve;
