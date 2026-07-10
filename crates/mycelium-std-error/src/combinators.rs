//! Pure combinators over `Option<T>` and `Result<T, E>` (spec §3).
//!
//! This module provides the errors-as-values ergonomics layer from `docs/spec/stdlib/error.md`:
//! every fallible op returns `Option`/`Result`; no combinator silently discards an error
//! (RFC-0014 I1 / RFC-0016 C1). The module is KC-3 (no new trusted code) — pure functions
//! over Rust's stdlib value sums, which serve as the Mycelium value model's `Option`/`Result`
//! in this Ring-2 Rust-first implementation.
//!
//! # Never-silent crux (C1 / I1)
//! Every combinator either:
//! - transforms the `Err`/`None` (it survives in the result sum), or
//! - re-propagates it (short-circuit or `?`), or
//! - explicitly recovers it (the caller must supply the recovery explicitly), or
//! - refuses loudly (`unwrap`/`expect`/`unwrap_err` — the named partial accessors).
//!
//! # EXPLAIN (C3)
//! The `ok` combinator is the one lossy conversion (FLAGGED per spec §7-Q2): `Err→None`
//! discards `ε`. It is documented prominently and is `EXPLAIN`-able in the guarantee
//! matrix. The partial accessors carry their diagnostic in the `RefusalRecord` (C3).
//! The `unwrap_or` family records its substitution via a `SubstitutionRecord` (C3/I2).
//!
//! # Guarantee tags (C2 / VR-5)
//! - All pure combinators: `Exact` (RFC-0016 C2 "len-style" case — pure value transforms).
//! - `unwrap_or` / `unwrap_or_else`: `Declared` (the substituted default is asserted, not
//!   proven — RFC-0014 I2). Downgrade to stay honest (VR-5).
//! - `recover` bridge: inherits the policy's honest tag; see spec §7-Q1 FLAG.

// ---- EXPLAIN / diagnostic support types --------------------------------------

/// The refusal record emitted when a named partial accessor (`unwrap`/`expect`/`unwrap_err`)
/// encounters the wrong variant. Carries the diagnostic as a message (C3 / spec §5/C3).
///
/// The *mechanism* (abort vs escalate vs `std.diag` record) is co-designed with M-510/M-520
/// (spec §7-Q3). This structure fixes the C3 obligation: the refusal is explicit and
/// carries information. In this Rust-first implementation we panic with this record's
/// message; the final form waits on `diag`/`recover` (FLAG Q3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusalRecord {
    /// The op name (`"unwrap"`, `"expect"`, or `"unwrap_err"`).
    pub op: &'static str,
    /// A caller-supplied reason (empty string for `unwrap`).
    pub reason: String,
    /// The variant that was actually present (for diagnostics).
    pub actual_variant: &'static str,
}

impl core::fmt::Display for RefusalRecord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.reason.is_empty() {
            write!(
                f,
                "{}: called on {} variant (explicit named partial — never a silent default)",
                self.op, self.actual_variant
            )
        } else {
            write!(
                f,
                "{}: {} (called on {} variant — explicit named partial)",
                self.op, self.reason, self.actual_variant
            )
        }
    }
}

/// The substitution record for `unwrap_or` / `unwrap_or_else`: records that a default was
/// substituted and that its guarantee is `Declared` (RFC-0014 I2 / VR-5). This is the C3
/// EXPLAIN artifact for the `Declared` rows in the guarantee matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstitutionRecord {
    /// The op name (`"unwrap_or"` or `"unwrap_or_else"`).
    pub op: &'static str,
    /// Guarantee tag — always `"Declared"` (a substituted default is asserted, not proven).
    pub guarantee_tag: &'static str,
}

// ---- transform (keep the sum shape; never collapse an error away) -----------

/// Map the `Ok`-side value; `Err` passes through unchanged (error preserved in the sum).
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `Err` is not touched; it survives in the output sum.
/// # Effects: none
#[inline]
pub fn map<T, U, E, F>(r: Result<T, E>, f: F) -> Result<U, E>
where
    F: FnOnce(T) -> U,
{
    r.map(f)
}

/// Map the `Err`-side value; `Ok` passes through unchanged.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `Ok` is not touched; `Err` is transformed (preserved in sum).
/// # Effects: none
#[inline]
pub fn map_err<T, E, F, G>(r: Result<T, E>, f: G) -> Result<T, F>
where
    G: FnOnce(E) -> F,
{
    r.map_err(f)
}

/// Monadic bind: apply `f` to the `Ok` value; `Err` short-circuits and **propagates**
/// (never dropped).
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): on `Err`, the error is propagated to the output — no drop.
/// # Effects: none (closure's effects are transparent to this combinator — C6)
#[inline]
pub fn and_then<T, U, E, F>(r: Result<T, E>, f: F) -> Result<U, E>
where
    F: FnOnce(T) -> Result<U, E>,
{
    r.and_then(f)
}

/// Explicit recovery hook: apply `f` to the `Err` value; `Ok` passes through.
/// `f` must return a `Result` — recover or re-propagate, never a drop.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `f` must yield a `Result`; there is no way to drop the error.
/// # Effects: none (closure's effects are transparent — C6)
#[inline]
pub fn or_else<T, E, F, G>(r: Result<T, E>, f: G) -> Result<T, F>
where
    G: FnOnce(E) -> Result<T, F>,
{
    r.or_else(f)
}

/// Filter an `Option`: `Some(x)` where `predicate(x)` is `false` becomes `None`.
/// This is a **typed transition** (named absence), not a silent loss.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `Some→None` via filter is a typed/explicit transition, not a drop.
/// # Effects: none (closure's effects are transparent — C6)
#[inline]
pub fn filter<T, F>(opt: Option<T>, predicate: F) -> Option<T>
where
    F: FnOnce(&T) -> bool,
{
    opt.filter(predicate)
}

/// Peek the `Ok` side with an effectful closure; the value and sum shape are **unchanged**.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `Err` propagates unchanged; this op is observational only.
/// # Effects: none from the combinator itself (closure may declare its own — C6)
#[inline]
pub fn inspect<T, E, F>(r: Result<T, E>, f: F) -> Result<T, E>
where
    F: FnOnce(&T),
{
    r.inspect(f)
}

/// Peek the `Err` side with an effectful closure; the value and propagation are **unchanged**.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): error propagation is not affected; this op is observational only.
/// # Effects: none from the combinator itself (closure may declare its own — C6)
#[inline]
pub fn inspect_err<T, E, F>(r: Result<T, E>, f: F) -> Result<T, E>
where
    F: FnOnce(&E),
{
    r.inspect_err(f)
}

// ---- convert between Option and Result (no information lost silently) -------

/// Convert `Option<T>` to `Result<T, E>` by naming the `None` case: `None → Err(err)`.
/// The absence is made explicit — never a silent drop.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `None` is explicitly named as `Err(err)`.
/// # Effects: none
#[inline]
pub fn ok_or<T, E>(opt: Option<T>, err: E) -> Result<T, E> {
    opt.ok_or(err)
}

/// Convert `Option<T>` to `Result<T, E>` with a lazily-computed error value.
/// `None → Err(f())`.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `None` is explicitly named as `Err(f())`.
/// # Effects: none from combinator itself (closure may declare its own — C6)
#[inline]
pub fn ok_or_else<T, E, F>(opt: Option<T>, f: F) -> Result<T, E>
where
    F: FnOnce() -> E,
{
    opt.ok_or_else(f)
}

/// Convert `Result<T, E>` to `Option<T>`: `Ok(t) → Some(t)`, `Err(e) → None`.
///
/// **FLAGGED LOSSY CONVERSION (spec §7-Q2 / C3).** This is the one combinator in
/// `std.error` that discards `ε` (`Err → None`). It is C1-honest only because it is
/// an **explicitly-named, EXPLAIN-able lossy conversion**, never an unflagged drop.
/// The spec FLAGs (Q2) whether it should require a name that makes the loss unmissable
/// (e.g. `ok_discarding_err`) — that decision awaits RFC-0016 §8-Q3 ratification.
///
/// The [`SubstitutionRecord`] is not returned here (the conversion is not a *default*
/// substitution) but the EXPLAIN obligation is discharged in the guarantee matrix row.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): the loss is explicit, flagged, and EXPLAIN-noted.
/// # Effects: none
#[inline]
pub fn ok<T, E>(r: Result<T, E>) -> Option<T> {
    r.ok()
}

/// Transpose `Option<Result<T, E>>` to `Result<Option<T>, E>`.
/// An `Err` inside the `Option` propagates out; no error is lost.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): `Err` inside the `Some` propagates to the outer `Result`.
/// # Effects: none
#[inline]
pub fn transpose<T, E>(opt: Option<Result<T, E>>) -> Result<Option<T>, E> {
    opt.transpose()
}

/// Flatten `Result<Result<T, E>, E>` to `Result<T, E>`.
/// The inner `Err` propagates to the outer; no wrapping is discarded silently.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): inner `Err` propagates; no error is dropped.
/// # Effects: none
#[inline]
pub fn flatten<T, E>(r: Result<Result<T, E>, E>) -> Result<T, E> {
    r.flatten()
}

/// Zip two `Option`s: both must be `Some`; either `None` short-circuits to `None`.
///
/// # Guarantee: `Exact` (pure combinator — RFC-0016 C2)
/// # Never-silent (I1/C1): either absent value yields `None` explicitly; nothing is lost.
/// # Effects: none
#[inline]
pub fn zip<A, B>(a: Option<A>, b: Option<B>) -> Option<(A, B)> {
    a.zip(b)
}

// ---- defaulted accessors (recover with an HONEST Declared tag) ---------------

/// Recover an `Err`/`None` with an explicitly-supplied default value.
/// Returns a `SubstitutionRecord` as the C3 EXPLAIN artifact.
///
/// # Guarantee: `Declared` — the substituted default is *asserted*, not proven (I2/VR-5).
/// # Never-silent (I1/C1): the recovery is explicit; the caller supplies the default.
/// # Effects: none
///
/// The returned `SubstitutionRecord` records the honest `Declared` tag for EXPLAIN (C3).
pub fn unwrap_or<T, E>(r: Result<T, E>, default: T) -> (T, SubstitutionRecord) {
    let record = SubstitutionRecord {
        op: "unwrap_or",
        guarantee_tag: "Declared",
    };
    (r.unwrap_or(default), record)
}

/// Recover an `Err`/`None` with a computed default from a closure.
/// Returns a `SubstitutionRecord` as the C3 EXPLAIN artifact.
///
/// # Guarantee: `Declared` — the substituted default is *asserted*, not proven (I2/VR-5).
/// # Never-silent (I1/C1): the recovery is explicit; the caller supplies the closure.
/// # Effects: none from this combinator (closure may declare its own — C6)
///
/// The returned `SubstitutionRecord` records the honest `Declared` tag for EXPLAIN (C3).
pub fn unwrap_or_else<T, E, F>(r: Result<T, E>, f: F) -> (T, SubstitutionRecord)
where
    F: FnOnce(E) -> T,
{
    let record = SubstitutionRecord {
        op: "unwrap_or_else",
        guarantee_tag: "Declared",
    };
    (r.unwrap_or_else(f), record)
}

/// Recover an `Option<T>` with an explicitly-supplied default value.
/// Returns a `SubstitutionRecord` as the C3 EXPLAIN artifact.
///
/// # Guarantee: `Declared` — the substituted default is *asserted*, not proven (I2/VR-5).
/// # Never-silent (I1/C1): the recovery is explicit; the caller supplies the default.
/// # Effects: none
pub fn unwrap_or_option<T>(opt: Option<T>, default: T) -> (T, SubstitutionRecord) {
    let record = SubstitutionRecord {
        op: "unwrap_or_option",
        guarantee_tag: "Declared",
    };
    (opt.unwrap_or(default), record)
}

/// Recover an `Option<T>` with a computed default from a closure.
/// Returns a `SubstitutionRecord` as the C3 EXPLAIN artifact.
///
/// # Guarantee: `Declared` — the substituted default is *asserted*, not proven (I2/VR-5).
/// # Never-silent (I1/C1): the recovery is explicit; the caller supplies the closure.
/// # Effects: none from this combinator (closure may declare its own — C6)
pub fn unwrap_or_else_option<T, F>(opt: Option<T>, f: F) -> (T, SubstitutionRecord)
where
    F: FnOnce() -> T,
{
    let record = SubstitutionRecord {
        op: "unwrap_or_else_option",
        guarantee_tag: "Declared",
    };
    (opt.unwrap_or_else(f), record)
}

// ---- explicit named partial accessors (refuse loudly; never a silent default) ---

/// Extract the `Ok` value.
///
/// # EXPLICIT NAMED PARTIAL (spec §3 / C1)
/// On `Err`, this **refuses loudly** (panics with a `RefusalRecord` message) and never
/// returns a silent default. This is the "I assert this is Ok" op — the one place a
/// computation can be intentionally stopped, and it does so audibly with a diagnostic.
///
/// The final refusal *mechanism* (abort vs escalate vs `std.diag` record) is FLAG Q3
/// in the spec; the guarantee — loud refusal, never silent — is fixed here.
///
/// # Guarantee: `Exact` (when it returns — the returned value is `Ok(t)`)
/// # Effects: none (on the `Ok` path)
///
/// # Panics
/// Panics if `r` is `Err`, with a message encoding the `RefusalRecord`.
pub fn unwrap<T, E: core::fmt::Debug>(r: Result<T, E>) -> T {
    match r {
        Ok(t) => t,
        Err(e) => {
            let record = RefusalRecord {
                op: "unwrap",
                reason: String::new(),
                actual_variant: "Err",
            };
            // FLAG Q3: the final mechanism is co-designed with M-510/M-520 (diag/recover).
            // This Rust-first implementation panics; the record is in the message.
            panic!("{record}: {e:?}");
        }
    }
}

/// Extract the `Ok` value with a caller-supplied reason for the expected state.
///
/// # EXPLICIT NAMED PARTIAL (spec §3 / C1)
/// On `Err`, **refuses loudly** with the caller-supplied `msg` in the diagnostic.
/// Same guarantee as `unwrap` — explicit, named, never silent.
///
/// # Guarantee: `Exact` (when it returns)
/// # Effects: none (on the `Ok` path)
///
/// # Panics
/// Panics if `r` is `Err`, with a message encoding the `RefusalRecord` + `msg`.
pub fn expect<T, E: core::fmt::Debug>(r: Result<T, E>, msg: &str) -> T {
    match r {
        Ok(t) => t,
        Err(e) => {
            let record = RefusalRecord {
                op: "expect",
                reason: msg.to_owned(),
                actual_variant: "Err",
            };
            // FLAG Q3: mechanism to be co-designed with M-510/M-520.
            panic!("{record}: {e:?}");
        }
    }
}

/// Extract the `Err` value: symmetric to `unwrap`.
///
/// # EXPLICIT NAMED PARTIAL (spec §3 / C1)
/// On `Ok`, **refuses loudly** with a diagnostic. The "I assert this is Err" op.
///
/// # Guarantee: `Exact` (when it returns)
/// # Effects: none (on the `Err` path)
///
/// # Panics
/// Panics if `r` is `Ok`, with a message encoding the `RefusalRecord`.
pub fn unwrap_err<T: core::fmt::Debug, E>(r: Result<T, E>) -> E {
    match r {
        Err(e) => e,
        Ok(t) => {
            let record = RefusalRecord {
                op: "unwrap_err",
                reason: String::new(),
                actual_variant: "Ok",
            };
            // FLAG Q3: mechanism to be co-designed with M-510/M-520.
            panic!("{record}: {t:?}");
        }
    }
}

// ---- RFC-0014 bridge (FLAG Q1 — RESOLVED) ------------------------------------
//
// The concrete `RecoverOutcome` shape, the `PolicyRef` resolution surface, and the declared/
// budgeted-effects handling are **owned by `std.recover` (M-520, RFC-0014)**, now landed. This
// module re-exports that surface at the crate root (`Outcome`/`Resolution`/`RecoverOutcome` =
// `Recovered | Propagated`, and `handle_classified`) rather than re-typing it — `std.error`
// adds no recovery algebra of its own (KC-3). The contract `std.error` stated abstractly holds
// verbatim in the landed types:
//   - the outcome is `Recovered | Propagated` with no drop variant (I1), and
//   - the tag is inherited from the policy, never laundered upward (I2/VR-5).
//
// The concrete recovery surface — `Outcome`/`Resolution`/`RecoverOutcome` and the
// `handle_classified` driver — lives in `std.recover` (M-520) and is re-exported from this
// crate's root (see `lib.rs`). `std.error` defines no recovery type or driver of its own (KC-3):
// it is the *bridge target* that hands an error value to `std.recover`, not the home of the
// recovery algebra. The contract `std.error` stated abstractly holds verbatim in the landed
// `std.recover` types — `Recovered | Propagated`, no drop variant (I1), tag inherited from the
// policy and never laundered upward (I2/VR-5).

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ---------------------------------------------------------------

    fn ok_42() -> Result<i32, &'static str> {
        Ok(42)
    }
    fn err_oops() -> Result<i32, &'static str> {
        Err("oops")
    }
    fn some_42() -> Option<i32> {
        Some(42)
    }
    fn none_i32() -> Option<i32> {
        None
    }

    // ---- map / map_err ---------------------------------------------------------

    /// `map` applies f to Ok; Err passes through unchanged (I1 / C1).
    /// Guard: a map that drops Err would pass Ok but this test catches it.
    #[test]
    fn map_transforms_ok_err_passes_through() {
        assert_eq!(map(ok_42(), |x| x * 2), Ok(84));
        assert_eq!(map(err_oops(), |x: i32| x * 2), Err("oops"));
    }

    /// `map_err` applies f to Err; Ok passes through unchanged.
    /// Guard: a map_err that drops Ok would pass Err but this test catches it.
    #[test]
    fn map_err_transforms_err_ok_passes_through() {
        assert_eq!(map_err(ok_42(), |e: &str| e.len()), Ok(42));
        assert_eq!(map_err(err_oops(), |e| e.len()), Err(4));
    }

    // ---- and_then / or_else ---------------------------------------------------

    /// `and_then` propagates Err without calling f (never-silent: Err is not dropped).
    /// Guard: calling f on Err would yield a wrong result; this test uses a side effect
    /// to confirm f was not called.
    #[test]
    fn and_then_short_circuits_on_err() {
        let called = core::cell::Cell::new(false);
        let result = and_then(err_oops(), |_x: i32| {
            called.set(true);
            Ok(0i32)
        });
        assert_eq!(result, Err("oops"), "Err must propagate");
        assert!(!called.get(), "f must not be called on Err (short-circuit)");
    }

    /// `and_then` applies f to Ok.
    #[test]
    fn and_then_applies_f_on_ok() {
        assert_eq!(and_then(ok_42(), |x| Ok::<_, &str>(x + 1)), Ok(43));
    }

    /// `or_else` applies f to Err; Ok passes through.
    #[test]
    fn or_else_applies_f_on_err_ok_passes_through() {
        assert_eq!(or_else(err_oops(), |_e| Ok::<i32, i32>(0)), Ok(0));
        assert_eq!(or_else(ok_42(), |_e: &str| Ok::<_, i32>(0)), Ok(42));
    }

    // ---- filter ----------------------------------------------------------------

    /// `filter` converts Some to None when predicate is false (typed transition, not drop).
    /// Guard: treating None as a dropped Some would fail a downstream use — no sentinel.
    #[test]
    fn filter_typed_transition_some_to_none() {
        assert_eq!(filter(some_42(), |&x| x > 100), None);
        assert_eq!(filter(some_42(), |&x| x > 0), Some(42));
        assert_eq!(filter(none_i32(), |&x| x > 0), None);
    }

    // ---- inspect / inspect_err -------------------------------------------------

    /// `inspect` peeks Ok without changing the sum.
    #[test]
    fn inspect_peeks_ok_sum_unchanged() {
        let mut seen = None;
        let r = inspect(ok_42(), |&x| seen = Some(x));
        assert_eq!(r, Ok(42));
        assert_eq!(seen, Some(42));
    }

    /// `inspect_err` peeks Err without changing propagation.
    #[test]
    fn inspect_err_peeks_err_propagation_unchanged() {
        let mut seen = None;
        let r = inspect_err(err_oops(), |&e| seen = Some(e));
        assert_eq!(r, Err("oops"));
        assert_eq!(seen, Some("oops"));
    }

    /// `inspect` on Err does NOT call f (Err propagates unchanged).
    #[test]
    fn inspect_on_err_does_not_call_f() {
        let called = core::cell::Cell::new(false);
        let r = inspect(err_oops(), |_| called.set(true));
        assert_eq!(r, Err("oops"));
        assert!(!called.get());
    }

    /// `inspect_err` on Ok does NOT call f (Ok propagates unchanged).
    #[test]
    fn inspect_err_on_ok_does_not_call_f() {
        let called = core::cell::Cell::new(false);
        let r = inspect_err(ok_42(), |_: &&str| called.set(true));
        assert_eq!(r, Ok(42));
        assert!(!called.get());
    }

    // ---- ok_or / ok_or_else ---------------------------------------------------

    /// `ok_or` names None as Err explicitly (never-silent).
    #[test]
    fn ok_or_names_none_as_err() {
        assert_eq!(ok_or(some_42(), "missing"), Ok(42));
        assert_eq!(ok_or(none_i32(), "missing"), Err("missing"));
    }

    /// `ok_or_else` names None as Err with a lazy error.
    #[test]
    fn ok_or_else_lazy_error_for_none() {
        let called = core::cell::Cell::new(false);
        let r = ok_or_else(some_42(), || {
            called.set(true);
            "missing"
        });
        assert_eq!(r, Ok(42));
        assert!(!called.get(), "f must not be called for Some");

        let r2 = ok_or_else(none_i32(), || "lazily-computed");
        assert_eq!(r2, Err("lazily-computed"));
    }

    // ---- ok (lossy, FLAGGED) --------------------------------------------------

    /// `ok` converts Ok(t) to Some(t) and Err to None.
    /// This is the one flagged lossy conversion (spec §7-Q2). Test confirms the conversion
    /// is explicit — the caller calls `ok` intentionally; there is no silent swallow.
    #[test]
    fn ok_lossy_conversion_explicit() {
        assert_eq!(ok(ok_42()), Some(42));
        assert_eq!(ok(err_oops()), None);
    }

    // ---- transpose / flatten / zip --------------------------------------------

    /// `transpose` lifts Err from inside Option to outside.
    #[test]
    fn transpose_propagates_err() {
        let inner_err: Option<Result<i32, &str>> = Some(Err("inner"));
        assert_eq!(transpose(inner_err), Err("inner"));
        assert_eq!(transpose(Some(Ok::<i32, &str>(1))), Ok(Some(1)));
        assert_eq!(transpose(None::<Result<i32, &str>>), Ok(None));
    }

    /// `flatten` propagates the inner Err.
    #[test]
    fn flatten_propagates_inner_err() {
        let inner: Result<Result<i32, &str>, &str> = Ok(Err("inner"));
        assert_eq!(flatten(inner), Err("inner"));
        assert_eq!(flatten(Ok(Ok::<i32, &str>(7))), Ok(7));
        assert_eq!(
            flatten(Err::<Result<i32, &str>, &str>("outer")),
            Err("outer")
        );
    }

    /// `zip` short-circuits to None when either is None.
    #[test]
    fn zip_short_circuits_on_none() {
        assert_eq!(zip(some_42(), Some("hi")), Some((42, "hi")));
        assert_eq!(zip(none_i32(), Some("hi")), None);
        assert_eq!(zip(some_42(), None::<&str>), None);
    }

    // ---- unwrap_or / unwrap_or_else (Declared) --------------------------------

    /// `unwrap_or` recovers Err with an explicitly supplied default; records Declared tag.
    #[test]
    fn unwrap_or_recovers_with_declared_tag() {
        let (val, record) = unwrap_or(ok_42(), 0);
        assert_eq!(val, 42, "Ok value must be returned");
        assert_eq!(record.guarantee_tag, "Declared");

        let (val2, record2) = unwrap_or(err_oops(), 99);
        assert_eq!(val2, 99, "default must be returned for Err");
        assert_eq!(record2.guarantee_tag, "Declared");
        assert_eq!(record2.op, "unwrap_or");
    }

    /// `unwrap_or_else` recovers Err with a computed default; records Declared tag.
    #[test]
    fn unwrap_or_else_recovers_with_declared_tag() {
        let (val, record) = unwrap_or_else(err_oops(), |_e| 99);
        assert_eq!(val, 99);
        assert_eq!(record.guarantee_tag, "Declared");
        assert_eq!(record.op, "unwrap_or_else");
    }

    /// The `Declared` tag is never `Exact` — downgrade is the rule (VR-5).
    /// Guard: changing guarantee_tag to "Exact" in the record makes this fail.
    #[test]
    fn unwrap_or_tag_is_not_exact() {
        let (_, record) = unwrap_or(err_oops(), 0);
        assert_ne!(
            record.guarantee_tag, "Exact",
            "unwrap_or must not claim Exact — the default is Declared (VR-5/I2)"
        );
    }

    // ---- unwrap / expect / unwrap_err (named partial) -------------------------

    /// `unwrap` on Ok returns the value.
    #[test]
    fn unwrap_on_ok_returns_value() {
        assert_eq!(unwrap(ok_42()), 42);
    }

    /// `unwrap` on Err panics (refuses loudly — never a silent default).
    /// Guard: returning a sentinel/default instead of panicking makes this fail.
    #[test]
    #[should_panic(expected = "unwrap")]
    fn unwrap_on_err_refuses_loudly() {
        unwrap(err_oops());
    }

    /// `expect` on Ok returns the value.
    #[test]
    fn expect_on_ok_returns_value() {
        assert_eq!(expect(ok_42(), "must be ok"), 42);
    }

    /// `expect` on Err panics with the caller-supplied message.
    /// Guard: a silent default would not panic with the caller message.
    #[test]
    #[should_panic(expected = "caller-reason")]
    fn expect_on_err_refuses_loudly_with_msg() {
        expect(err_oops(), "caller-reason");
    }

    /// `unwrap_err` on Err returns the error.
    #[test]
    fn unwrap_err_on_err_returns_error() {
        assert_eq!(unwrap_err(err_oops()), "oops");
    }

    /// `unwrap_err` on Ok panics (refuses loudly — symmetric partial).
    /// Guard: returning a sentinel instead of panicking makes this fail.
    #[test]
    #[should_panic(expected = "unwrap_err")]
    fn unwrap_err_on_ok_refuses_loudly() {
        unwrap_err(ok_42());
    }

    // The recover bridge's behaviour (Ok→floor, Err→propagate, no drop variant, budgeted
    // effects) is owned and tested by `std.recover` (M-520); `std.error` only re-exports the
    // surface, so its tests live there — not duplicated here.

    // ---- property tests: lattice + honesty bounds ----------------------------

    /// Property: `map` is the identity on `Err` — for all error shapes, Err passes through.
    /// This is the per-op "never-silent" property bound for `map` (spec §4).
    #[test]
    fn property_map_never_drops_err() {
        // Exhaustive for a small error domain (the spec's "property test for every bound").
        for &e in &["a", "b", "long error", "", "oops"] {
            let r: Result<i32, &str> = Err(e);
            let out = map(r, |x| x * 100);
            assert_eq!(out, Err(e), "map must preserve Err: {e:?}");
        }
    }

    /// Property: `map_err` is the identity on `Ok` — for all ok values, Ok passes through.
    #[test]
    fn property_map_err_never_drops_ok() {
        for &v in &[0i32, 1, -1, i32::MAX, i32::MIN] {
            let r: Result<i32, &str> = Ok(v);
            let out = map_err(r, |e: &str| e.len());
            assert_eq!(out, Ok(v), "map_err must preserve Ok: {v}");
        }
    }

    /// Property: `and_then` propagates Err for all error values (never-silent bound).
    #[test]
    fn property_and_then_propagates_all_errs() {
        for &e in &["e1", "e2", "error3"] {
            let r: Result<i32, &str> = Err(e);
            let out = and_then(r, |_| Ok::<i32, &str>(999));
            assert_eq!(out, Err(e), "and_then must propagate Err: {e:?}");
        }
    }

    /// Property: `or_else` passes through Ok for all ok values.
    #[test]
    fn property_or_else_passes_through_ok() {
        for &v in &[0i32, 42, -99] {
            let r: Result<i32, &str> = Ok(v);
            let out = or_else(r, |_| Ok::<_, i32>(0));
            assert_eq!(out, Ok(v), "or_else must pass through Ok: {v}");
        }
    }

    /// Property: `filter` typed transition — `filter(None, _)` is always None.
    #[test]
    fn property_filter_none_is_always_none() {
        for pred_result in &[true, false] {
            let out = filter(none_i32(), |_| *pred_result);
            assert_eq!(out, None, "filter(None, _) must be None");
        }
    }

    /// Property: `ok_or` never returns Ok for None (naming the absence explicitly).
    #[test]
    fn property_ok_or_names_none_as_err() {
        for &e in &["e1", "e2", "missing"] {
            let out: Result<i32, &str> = ok_or(none_i32(), e);
            assert!(out.is_err(), "ok_or(None, e) must be Err for any e");
            assert_eq!(out.unwrap_err(), e);
        }
    }

    /// Property: `unwrap_or` result for Err is exactly the supplied default (Declared).
    #[test]
    fn property_unwrap_or_exact_default_for_err() {
        for &default in &[0i32, 1, 99, -1, i32::MIN] {
            let r: Result<i32, &str> = Err("e");
            let (val, record) = unwrap_or(r, default);
            assert_eq!(
                val, default,
                "unwrap_or must return exactly the default for Err"
            );
            assert_eq!(
                record.guarantee_tag, "Declared",
                "tag must be Declared (VR-5)"
            );
        }
    }

    // The `recover`-never-drops property (Outcome is always `Recovered | Propagated`) is owned
    // and property-tested by `std.recover` (M-520) over its real `Outcome`/`Resolution` types;
    // `std.error` re-exports that surface and does not re-test it here.

    /// Property: `zip` always yields None when either input is None.
    #[test]
    fn property_zip_none_absorbs() {
        for a in [some_42(), none_i32()] {
            let out = zip(a, None::<i32>);
            assert_eq!(out, None, "zip with None must be None");
        }
    }

    /// Property: `transpose` preserves the Err inside Some.
    #[test]
    fn property_transpose_preserves_inner_err() {
        for &e in &["e1", "e2", "err"] {
            let inner: Option<Result<i32, &str>> = Some(Err(e));
            assert_eq!(transpose(inner), Err(e));
        }
    }

    /// Property: `flatten` propagates both inner and outer Errs.
    #[test]
    fn property_flatten_propagates_all_errs() {
        let errs = ["outer", "inner"];
        let inner: Result<Result<i32, &str>, &str> = Ok(Err(errs[1]));
        assert_eq!(flatten(inner), Err("inner"));
        let outer: Result<Result<i32, &str>, &str> = Err(errs[0]);
        assert_eq!(flatten(outer), Err("outer"));
    }

    // ---- SubstitutionRecord / RefusalRecord structural tests ------------------

    /// SubstitutionRecord carries the honest `Declared` tag.
    #[test]
    fn substitution_record_carries_declared() {
        let r = SubstitutionRecord {
            op: "unwrap_or",
            guarantee_tag: "Declared",
        };
        assert_eq!(r.guarantee_tag, "Declared");
    }

    /// RefusalRecord Display includes op name (C3 diagnostic).
    #[test]
    fn refusal_record_display_includes_op() {
        let r = RefusalRecord {
            op: "expect",
            reason: "should have been ok".to_owned(),
            actual_variant: "Err",
        };
        let msg = r.to_string();
        assert!(msg.contains("expect"), "display must include op name");
        assert!(
            msg.contains("should have been ok"),
            "display must include reason"
        );
        assert!(msg.contains("Err"), "display must include actual variant");
    }
}
