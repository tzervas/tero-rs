//! `std.core::error_scaffold` — the shared, non-coupling scaffold for stdlib error
//! types (M-535, E5-1; DN-17 §2.4/§4 P3).
//!
//! # What this is (and deliberately is *not*)
//! Every `mycelium-std-*` crate hand-rolls, per error type, the *same three mechanical
//! pieces*: `#[derive(Debug, …)]`, an `impl std::error::Error` marker (occasionally with a
//! `source()` delegate), and a `*_is_std_error` compile-time test. DN-17 §2.4 measured
//! ~600 LOC of this `Display`/`Error`/test boilerplate as structurally repeated. This
//! module factors out **only the mechanical part**:
//!
//! - [`StdError`] — a *marker* super-trait (`Debug + Display + std::error::Error`) that
//!   *names* the per-op never-silent / EXPLAIN contract (RFC-0016 §4.1 C1/C3) so "this
//!   value is an honest stdlib error" is inspectable and bound-checkable, with a blanket
//!   impl so no type opts in by hand.
//! - [`impl_std_error!`](crate::impl_std_error) — a tiny declarative macro that emits the boilerplate
//!   `impl std::error::Error for T {}` (optionally with a `source()` arm), leaving the
//!   **hand-written, domain-specific `Display` message in the caller** — untouched.
//! - `assert_is_std_error` / `assert_display_contains` — the test helpers DN-17 §4
//!   blesses as the *safe* consolidation (the `*_is_std_error` + `*_display_includes_*`
//!   duplication), gated behind `#[cfg(any(test, feature = "test-support"))]` (so they are
//!   absent — and these names unlinkable — in a default doc build).
//!
//! ## What it does NOT do (VR-5 / DN-17 §5 — non-coupling by construction)
//! It never generates, alters, or inspects a `Display` *message*; never adds, removes, or
//! renames an error variant; never changes a `#[derive]`; and never touches a per-module
//! guarantee tag or the per-module guarantee matrix (RFC-0016 §4.5 — intentionally
//! per-module, DN-17 §3). A refactor through this scaffold is **behaviour-preserving**: the
//! only code it emits is the empty `Error` marker and an optional `source()` delegate, both
//! of which are byte-identical everywhere they appear today. The per-module *honesty
//! contracts stay decoupled* — this is the line DN-17 §5 draws, and the scaffold respects it.
//!
//! # Guarantee tag
//! **`Exact`** — the scaffold is pure, total, allocation-free glue: [`StdError`] is a marker
//! with a trivial blanket impl, [`impl_std_error!`](crate::impl_std_error) expands to the exact trait impl the
//! caller would otherwise write, and the assert helpers are total predicates over a borrowed
//! error. No approximation, no selection, no representation choice is introduced (C5: it adds
//! no trusted code — it consumes `std::error::Error`).
//!
//! # Grounding
//! DN-17 §2.4 (the survey), §4 P3 (test-helper = safe; macro = post-ratification), §5 (the
//! non-coupling caveat); RFC-0016 §4.1 C1 (never-silent), C3 (EXPLAIN / no black box), C5
//! (above the kernel); G2 (never-silent); VR-5 (a refactor never changes a guarantee).

use core::fmt;

/// Marker super-trait: *this value is an honest stdlib error*.
///
/// A type is an [`StdError`] iff it is `Debug + Display + std::error::Error` — i.e. it can be
/// shown to a human (`Display`), inspected by a developer (`Debug`), and chained as a cause
/// (`std::error::Error::source`). The blanket impl below means **every** type already meeting
/// that bound *is* an `StdError` with no per-type opt-in; the trait exists to give the
/// contract a *name* the rest of the library (and tests) can bound on:
///
/// ```
/// use mycelium_std_core::error_scaffold::StdError;
/// fn surface<E: StdError>(e: &E) -> String { e.to_string() }
/// ```
///
/// # Contract named (not enforced-by-types — RFC-0016 §4.1)
/// - **C1 / G2 (never-silent).** An `StdError` is a *value*: producing one is the explicit,
///   non-sentinel signal that an operation refused. The scaffold cannot enforce that a crate
///   *returns* it rather than clamping — that is each op's obligation — but it gives the
///   refusal a uniform, honest type.
/// - **C3 (EXPLAIN / no black box).** `Display` carries the human-legible *why*; `source()`
///   carries the machine-walkable cause chain. An `Err` is as inspectable as an `Ok`.
///
/// # Guarantee tag: `Exact` (a marker with a trivial blanket impl — no behaviour of its own).
pub trait StdError: fmt::Debug + fmt::Display + std::error::Error {}

// Blanket impl: anything already satisfying the bound is an `StdError`. This is why no
// existing error type needs to opt in by hand (DRY: the contract is named once, here).
impl<T: ?Sized + fmt::Debug + fmt::Display + std::error::Error> StdError for T {}

/// Emit the mechanical `impl std::error::Error` marker for an error type, optionally with a
/// `source()` delegate — and **nothing else**.
///
/// This is the boilerplate every `mycelium-std-*/src/error.rs` repeats verbatim. The caller
/// keeps its own `#[derive(Debug, …)]` and its own hand-written `impl Display` (the
/// domain-specific message is *intentionally* not automated — DN-17 §3/§5); the macro only
/// removes the empty / mechanically-identical `impl std::error::Error`.
///
/// # Forms
/// ```text
/// impl_std_error!(MyError);                        // -> impl std::error::Error for MyError {}
/// impl_std_error!(MyError<T>, generics = [T: Debug], where = [T: Debug]);  // generic
/// impl_std_error!(MyError, source = |this| { match this { … } });          // source() delegate
/// ```
///
/// The `source = |this| …` arm receives `&self` as `this` and must return
/// `Option<&(dyn std::error::Error + 'static)>` — exactly the body the caller would write.
///
/// # Why a macro and not a `#[derive]`
/// KC-3 / KISS: a `proc-macro` would add a build-time dependency and a separate crate for a
/// two-line impl. A `macro_rules!` keeps the dependency floor at zero and the expansion
/// trivially auditable (`cargo expand` shows the exact impl). It is **opt-in**: a crate may
/// keep writing the impl by hand and nothing breaks.
///
/// # Guarantee tag: `Exact` — expands to the precise impl the caller would otherwise write.
#[macro_export]
macro_rules! impl_std_error {
    // Plain marker: `impl std::error::Error for Ty {}`.
    ($ty:ty) => {
        impl ::std::error::Error for $ty {}
    };

    // Generic marker with optional where-clause: `impl<…> Error for Ty<…> where … {}`.
    ($ty:ty, generics = [$($g:tt)*] $(, where = [$($w:tt)*])? $(,)?) => {
        impl<$($g)*> ::std::error::Error for $ty $(where $($w)*)? {}
    };

    // Marker with a `source()` delegate. `$this` binds `&self`; the block returns
    // `Option<&(dyn std::error::Error + 'static)>`.
    ($ty:ty, source = |$this:ident| $body:block $(,)?) => {
        impl ::std::error::Error for $ty {
            fn source(&self) -> ::core::option::Option<&(dyn ::std::error::Error + 'static)> {
                let $this = self;
                $body
            }
        }
    };
}

/// Compile-and-run assertion that `e` satisfies the `std::error::Error` (and thus [`StdError`])
/// contract — the consolidation of the `*_is_std_error` test every error file repeats.
///
/// It coerces `e` to `&dyn std::error::Error` (the compile-time half, exactly the existing
/// `let _: &dyn std::error::Error = &e;` line) and exercises `Display` + `Debug` once (the
/// run-time half — that neither panics and `Display` is non-empty), then returns the rendered
/// `Display` string so a caller can fold the message check in the same call.
///
/// ```
/// # #[cfg(any(test, feature = "test-support"))]
/// # {
/// use mycelium_std_core::error_scaffold::assert_is_std_error;
/// #[derive(Debug)]
/// struct E;
/// impl std::fmt::Display for E {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("boom") }
/// }
/// impl std::error::Error for E {}
/// let shown = assert_is_std_error(&E);
/// assert_eq!(shown, "boom");
/// # }
/// ```
///
/// # Guarantee tag: `Exact` — a total predicate over a borrowed error; no approximation.
#[cfg(any(test, feature = "test-support"))]
pub fn assert_is_std_error<E: StdError>(e: &E) -> String {
    // Compile-time: the bound `E: StdError` already requires `std::error::Error`. This
    // coercion mirrors the historical `let _: &dyn std::error::Error = &e;` guard exactly.
    let dynamic: &dyn std::error::Error = e;
    // Run-time: `Debug` and `Display` must both render without panicking, and the human
    // message must be non-empty (C3 — an error that displays as "" carries no EXPLAIN).
    let _ = format!("{dynamic:?}");
    let shown = dynamic.to_string();
    assert!(
        !shown.is_empty(),
        "an StdError must render a non-empty Display message (C3 / EXPLAIN): {dynamic:?}"
    );
    shown
}

/// Assert that an error's `Display` contains every `needle` — the consolidation of the
/// near-identical `*_display_includes_*_fields` guard tests (each currently a hand-written
/// run of `assert!(s.contains(..))`).
///
/// Panics with a message naming the missing needle and the full rendered error, so a failing
/// guard points straight at the dropped field.
///
/// ```
/// # #[cfg(any(test, feature = "test-support"))]
/// # {
/// use mycelium_std_core::error_scaffold::assert_display_contains;
/// #[derive(Debug)]
/// struct E { index: usize, len: usize }
/// impl std::fmt::Display for E {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         write!(f, "oob index={} len={}", self.index, self.len)
///     }
/// }
/// impl std::error::Error for E {}
/// assert_display_contains(&E { index: 5, len: 3 }, &["5", "3"]);
/// # }
/// ```
///
/// # Guarantee tag: `Exact` — a total substring predicate; no approximation.
#[cfg(any(test, feature = "test-support"))]
pub fn assert_display_contains<E: StdError>(e: &E, needles: &[&str]) {
    let shown = e.to_string();
    for needle in needles {
        assert!(
            shown.contains(needle),
            "Display of {e:?} must contain {needle:?} (field would be silently dropped); \
             got {shown:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A representative hand-written error in the exact shape the std crates use: a
    // `#[derive(Debug)]` enum, a hand-written `match`-on-`self` `Display`, and the
    // mechanical `Error` impl supplied by the scaffold macro.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum SampleErr {
        IndexOob { index: usize, len: usize },
        Refused { why: String },
    }

    impl fmt::Display for SampleErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                SampleErr::IndexOob { index, len } => {
                    write!(f, "index out of bounds (index={index}, len={len})")
                }
                SampleErr::Refused { why } => write!(f, "refused: {why}"),
            }
        }
    }

    // Macro form 1: plain marker.
    impl_std_error!(SampleErr);

    // A wrapper exercising the `source =` form (the io/recover chaining pattern).
    #[derive(Debug)]
    struct Wrapper {
        inner: SampleErr,
    }

    impl fmt::Display for Wrapper {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "wrapper: {}", self.inner)
        }
    }

    // Macro form 3: marker + source() delegate.
    impl_std_error!(Wrapper, source = |this| { Some(&this.inner) });

    // A generic error exercising the `generics =` / `where =` form (the std-cmp pattern).
    #[derive(Debug)]
    struct Narrowed<T: fmt::Debug> {
        rejected: T,
    }

    impl<T: fmt::Debug> fmt::Display for Narrowed<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "not representable: {:?}", self.rejected)
        }
    }

    // Macro form 2: generic marker with a bound.
    impl_std_error!(Narrowed<T>, generics = [T: fmt::Debug], where = [T: fmt::Debug]);

    /// The blanket impl makes any `Debug + Display + Error` type an `StdError` — no opt-in.
    /// Guard: removing the blanket impl makes this fail to compile.
    #[test]
    fn blanket_impl_covers_conforming_types() {
        fn require_std_error<E: StdError>(_: &E) {}
        require_std_error(&SampleErr::IndexOob { index: 1, len: 0 });
        require_std_error(&Wrapper {
            inner: SampleErr::Refused { why: "x".into() },
        });
        require_std_error(&Narrowed { rejected: 7u8 });
    }

    /// The plain macro form yields a working `std::error::Error` whose `source()` is `None`.
    /// Guard: a macro that emitted a non-trivial `source()` here would fail.
    #[test]
    fn plain_macro_yields_std_error_without_source() {
        let e = SampleErr::Refused {
            why: "consumed".into(),
        };
        let dynamic: &dyn std::error::Error = &e;
        assert!(
            std::error::Error::source(dynamic).is_none(),
            "plain impl_std_error! must not invent a source()"
        );
    }

    /// The `source =` macro form chains to the inner error (the io/recover pattern).
    /// Guard: dropping the source arm makes `source()` return None and this fails.
    #[test]
    fn source_macro_form_chains_cause() {
        let w = Wrapper {
            inner: SampleErr::IndexOob { index: 9, len: 3 },
        };
        let src = std::error::Error::source(&w).expect("source() must chain to inner");
        assert!(
            src.to_string().contains("index out of bounds"),
            "source() must expose the inner error's Display; got {src}"
        );
    }

    /// `assert_is_std_error` returns the rendered Display and rejects an empty message.
    #[test]
    fn assert_is_std_error_returns_display() {
        let shown = assert_is_std_error(&SampleErr::IndexOob { index: 5, len: 3 });
        assert!(shown.contains("index out of bounds"));
    }

    /// `assert_display_contains` passes when all needles are present.
    #[test]
    fn assert_display_contains_accepts_present_needles() {
        assert_display_contains(
            &SampleErr::IndexOob { index: 5, len: 3 },
            &["index out of bounds", "5", "3"],
        );
    }

    /// `assert_display_contains` panics when a needle is missing (a dropped field).
    /// Guard: this is the failure mode the helper exists to catch.
    #[test]
    #[should_panic(expected = "must contain")]
    fn assert_display_contains_rejects_missing_needle() {
        assert_display_contains(&SampleErr::Refused { why: "x".into() }, &["NOT-PRESENT"]);
    }

    /// The macro does not alter the hand-written Display message in any way (behaviour-
    /// preserving — the whole point). Guard: any message rewrite by the macro fails here.
    #[test]
    fn macro_preserves_handwritten_display_verbatim() {
        let e = SampleErr::IndexOob { index: 2, len: 1 };
        assert_eq!(e.to_string(), "index out of bounds (index=2, len=1)");
    }
}
