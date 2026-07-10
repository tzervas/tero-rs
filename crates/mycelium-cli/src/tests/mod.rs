// In-crate test module for `mycelium-cli` (M-797 layout; CLAUDE.md §test-layout).
// Logic files carry no `#[cfg(test)]` blocks — tests live here, one submodule per source module.
mod lib_root;
mod stream;
mod unbounded;
