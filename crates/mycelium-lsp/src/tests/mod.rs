//! In-crate white-box unit tests (M-797 as-touched): relocated from the logic files'
//! inline `#[cfg(test)] mod tests` so the logic files carry no test code. Each submodule
//! uses `use crate::<mod>::*` for white-box access to its sibling logic module's private items.

mod completions;
mod lint;
mod project;
mod semantic;
