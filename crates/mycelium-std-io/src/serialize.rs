//! Serialize / deserialize surface (spec §3 — the serialize half).
//!
//! Two format entry points ([`Format`]) over `mycelium-core`'s existing `serde`
//! implementation (M-104):
//!
//! - **`Wire`** — the self-describing `[Repr] ‖ [Meta] ‖ [payload]` binary-JSON
//!   form of RFC-0001 §4.8 (schema-travels-with-data, faithfully round-trippable
//!   including `Meta`).
//! - **`Json`** — the **one canonical JSON projection** (`fmt.to_json` delegates
//!   here — README §5 seam; spec §7-Q1 FLAGGED); the same serde grammar used in
//!   the `Wire` form, rendered as a compact UTF-8 text object.
//!
//! The round-trip property `deserialize(serialize(v, f), f) ≡ v` including `Meta`
//! is asserted as a **property test** (proptest) in the in-crate test module
//! (`src/tests/serialize.rs`).  The tag is **`Empirical`** — not `Proven` — because no
//! side-condition theorem has been checked for this implementation (VR-5 / spec §4.2 Q2).
//!
//! # Honesty stance
//! - `serialize`/`to_json` are **fallible**: they project every `Value` whose payload is
//!   JSON-representable (RFC-0001 §4.8) and **refuse** a `Value` carrying a non-finite `f64`
//!   (`NaN`/`±∞`) with `Err(SerError::OutOfDomain)`. JSON has no non-finite literal, and
//!   `serde_json` would silently emit `null` — a lossy, ambiguous encoding (`NaN` and `±∞` both
//!   collapse to `null`, breaking the round-trip and colliding identity). Refusing is never-silent
//!   (C1/G2). They borrow a `&Value` and never mutate or re-key it (C4 — projection, not identity;
//!   ADR-003).
//! - `deserialize`/`from_json` return `Err(SerError)` with a **locus** on any decode
//!   failure — never a partially-filled `Value` or a zeroed sentinel (C1/G2).
//!
//! # C5 / no new trusted code
//! This module wraps `mycelium-core`'s `serde::{Serialize, Deserialize}` for
//! `Value` (landed M-104).  It adds **no** new serialization logic of its own;
//! `serde_json` is the only dependency beyond `mycelium-core` (KC-3).
//!
//! # FLAG: §8-Q6 — no `wild`/FFI here
//! The serialize half is purely in-memory (`Vec<u8>` / `String`); it uses no OS
//! facilities.  The io half (see `io.rs`) defers its OS floor to `std-sys` (M-541).

use mycelium_core::value::Payload;
use mycelium_core::Value;

use crate::error::{ByteOffset, FieldPath, SerError};

// ── Format selector ──────────────────────────────────────────────────────────

/// The two supported serialization formats (spec §3).
///
/// Both formats share the same self-describing grammar (`[Repr] ‖ [Meta] ‖
/// [payload]`), so the round-trip property holds for each independently.  The
/// only difference is the byte representation: `Wire` is JSON-in-bytes (the
/// same substrate used for the `Value` serde form in M-104), `Json` is the
/// UTF-8 text form suitable for human/tool consumption (G11 dual projection).
///
/// # EXPLAIN-able selection
/// A `Format` value is the reified, inspectable selection artifact (C3): the
/// choice of `Wire` vs `Json` is visible at every call site — there is no
/// ambient default that silently changes the wire form (spec §7-Q5 / RFC-0016
/// §8-Q3 tension A; required-explicit until the per-ring ergonomics pass M-540).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// The self-describing `[Repr] ‖ [Meta] ‖ [payload]` binary-JSON form
    /// (RFC-0001 §4.8).  The byte representation is a compact JSON object
    /// encoded as UTF-8 bytes (not a distinct binary encoding — see FLAG §8-Q6
    /// for a future binary wire form; the grammar is identical to `Json` but
    /// the container is `Vec<u8>`).
    Wire,
    /// The **one canonical JSON projection** — compact UTF-8 JSON text.
    ///
    /// `fmt.to_json` (M-533) **delegates** to this format (README §5 seam):
    /// one projection, two entry points.  The round-trip property is
    /// established here and shared — not duplicated.  (Delegation is FLAGGED
    /// §7-Q1 pending maintainer sign-off.)
    Json,
}

// ── serialize / deserialize (Wire and Json) ──────────────────────────────────

/// A `Value` carrying a non-finite `f64` (`NaN`/`±∞`) in a `Dense`/`Vsa` payload has no faithful
/// JSON representation (`serde_json` would silently emit `null`), so it is refused here.
///
/// Returns `Err(SerError::OutOfDomain)` naming the payload index of the first non-finite scalar;
/// `Ok(())` when every scalar is finite (or the payload has no `f64`).
fn check_json_representable(v: &Value) -> Result<(), SerError> {
    let scalars: &[f64] = match v.payload() {
        Payload::Scalars(s) | Payload::Hypervector(s) => s,
        Payload::Bits(_) | Payload::Trits(_) => return Ok(()),
        // A sequence (RFC-0032 D3) has no flat f64 payload, but its elements might — recurse so a
        // non-finite scalar nested inside a `Seq` is still caught, never silently emitted as null
        // (G2). The first offending element propagates its `OutOfDomain` up.
        Payload::Seq(elems) => {
            for e in elems {
                check_json_representable(e)?;
            }
            return Ok(());
        }
        // A byte string (RFC-0032 D4) carries no f64 — always JSON-representable here.
        Payload::Bytes(_) => return Ok(()),
        // A scalar float (ADR-040; M-896) is always JSON-representable: its wire form is a
        // *string* ("1.5"/"-0.0"/"inf"/"-inf"/"NaN" — mycelium-core `PayloadWire::Float`), so the
        // in-band IEEE specials ride the wire faithfully; nothing to refuse here (unlike the
        // number-array `Scalars`/`Hypervector` forms above).
        Payload::Float(_) => return Ok(()),
    };
    if let Some(pos) = scalars.iter().position(|x| !x.is_finite()) {
        return Err(SerError::OutOfDomain {
            path: FieldPath::from_static("payload"),
            why: format!(
                "non-finite f64 at payload index {pos} has no JSON representation \
                 (serde_json would silently emit null, losing NaN/±∞ and colliding identity); \
                 refused — never-silent (C1/G2)"
            ),
        });
    }
    Ok(())
}

/// Project `v` to the wire/JSON byte form for the given `format`.
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// A faithful projection: every JSON-representable `Value` has a wire/JSON form (RFC-0001 §4.8).
/// `serialize` borrows `v` immutably; it never mutates or re-keys the value, so the content hash
/// is unchanged (C4/ADR-003).
///
/// # Fallibility: `Err(SerError::OutOfDomain)`
/// A `Value` carrying a non-finite `f64` (`NaN`/`±∞`) is refused — JSON cannot represent it and
/// `serde_json` would silently emit `null` (a lossy, identity-colliding encoding). Never-silent
/// (C1/G2). Every other well-formed `Value` serializes (M-104's `serde` impl is total over the
/// finite domain).
///
/// # Effects: none
/// Pure computation over the in-memory value; no IO.
///
/// # EXPLAIN-able: n/a
/// A faithful projection has no hidden selection or approximation (spec §4/C3).
pub fn serialize(v: &Value, format: Format) -> Result<Vec<u8>, SerError> {
    check_json_representable(v)?;
    // After the finiteness check, `Value`'s serde impl is total — the only way `to_vec` could
    // error is a non-finite float (excluded) or an I/O error on the in-memory writer (impossible).
    let bytes = match format {
        // Wire and Json share the same grammar; the `Format` tag is preserved at the call site
        // (C3 — reified selection), so the two arms intentionally produce identical bytes.
        Format::Wire | Format::Json => {
            serde_json::to_vec(v).expect("Value serialization is total over finite Values (M-104)")
        }
    };
    Ok(bytes)
}

/// Recover a `Value` from `bytes` serialized in the given `format`.
///
/// # Guarantee tag: `Empirical` (round-trip property; spec §4.2)
/// `deserialize(serialize(v, f), f) ≡ v` holds over a generated property-test
/// corpus (asserted in `#[cfg(test)]`).  The tag is **`Empirical`** — not
/// `Proven` — because no injectivity/totality theorem over the closed grammar
/// has been checked here (VR-5 / spec §7-Q2).
///
/// # Fallibility: `Err(SerError)` with a locus (C1 — never-silent)
/// Any of the five failure modes below; the error carries the **byte offset or
/// field path** of the failure (RFC-0013 I1):
///
/// - `Truncated{at}` — input ended before a complete value was decoded.
/// - `Malformed{at, why}` — bytes do not parse (grammar violation).
/// - `UnknownTag{path, tag}` — unrecognized `Repr`/ctor/`Meta` tag.
/// - `OutOfDomain{path, why}` — field decodes but violates a value-model
///   invariant (e.g. payload length ≠ repr width).
/// - `BudgetExceeded{kind}` — a declared decode budget overrun (ADR-015).
///
/// **No partially-filled `Value` is ever returned** (C1/G2).
///
/// # Effects: none
/// Pure over the byte input; no IO.
pub fn deserialize(bytes: &[u8], _format: Format) -> Result<Value, SerError> {
    // Delegate to serde_json / mycelium-core's Value deserializer.
    // Map serde errors to the typed SerError variants with locus information
    // extracted from the serde_json error (byte offset is available via
    // `serde_json::Error::offset()`; classification follows the error category).
    serde_json::from_slice::<Value>(bytes).map_err(|e| map_serde_error(e, bytes))
}

// ── Canonical JSON entry points ───────────────────────────────────────────────

/// The **one canonical JSON projection**: project `v` to compact UTF-8 JSON text.
///
/// `fmt.to_json` (M-533) **delegates** to this function (README §5 seam; spec
/// §7-Q1 FLAGGED pending maintainer sign-off).  The round-trip property
/// `from_json(to_json(v)) ≡ v` is established here once and not duplicated.
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// Faithful projection (identical to `serialize(v, Format::Json)` as `String`).
///
/// # Fallibility: `Err(SerError::OutOfDomain)`
/// Refuses a `Value` carrying a non-finite `f64` (same domain as [`serialize`]) — JSON cannot
/// represent it and `serde_json` would silently emit `null` (C1/G2). Total over the finite domain.
///
/// # Effects: none
pub fn to_json(v: &Value) -> Result<String, SerError> {
    check_json_representable(v)?;
    Ok(serde_json::to_string(v)
        .expect("to_json is total over finite Values (M-104; non-finite excluded above)"))
}

/// Recover a `Value` from canonical JSON text.
///
/// `from_json(to_json(v)) ≡ v` (the round-trip property, `Empirical`).
///
/// # Guarantee tag: `Empirical` (round-trip; spec §4.2)
///
/// # Fallibility: `Err(SerError)` with a locus (C1 — never-silent)
///
/// # Effects: none
pub fn from_json(text: &str) -> Result<Value, SerError> {
    serde_json::from_str::<Value>(text).map_err(|e| map_serde_error_str(e, text))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Map a `serde_json::Error` (from `from_slice`) to a typed [`SerError`] with
/// the best available locus information.
///
/// `serde_json::Error` exposes `line()` and `column()` (1-based); we convert to
/// a byte-offset approximation using the line/column and the raw input (best-effort
/// — a precise byte offset would require a byte-counting deserializer, which is
/// deferred to a future codec improvement; this is the honest `Empirical` floor).
fn map_serde_error(e: serde_json::Error, input: &[u8]) -> SerError {
    // Approximate the byte offset from line/column.  serde_json provides
    // `line()` (1-based) and `column()` (1-based); we walk the input to find the
    // byte position of (line, column).  If the input is empty or line exceeds the
    // line count, fall back to `input.len()` (end-of-input, indicating truncation).
    let byte_offset = ByteOffset(approx_byte_offset(input, e.line(), e.column()));

    // serde_json classifies errors as Io (won't occur for in-memory slices),
    // Syntax (malformed JSON), Data (type mismatch / value invariant), or Eof.
    // We map those to our typed variants.
    let msg = e.to_string();

    if is_truncated_error_msg(&msg, input) {
        SerError::Truncated { at: byte_offset }
    } else if is_unknown_tag_error(&msg) {
        // Check unknown-tag BEFORE the domain heuristic: serde's "unknown variant `x`, expected
        // one of [`repr`, …]" message contains the literal "repr", which the substring-based
        // `is_domain_error` would otherwise misclassify as OutOfDomain (wrong variant — C1/C3).
        SerError::UnknownTag {
            path: FieldPath::from_static("repr"),
            tag: extract_unknown_tag(&msg),
        }
    } else if is_domain_error(&msg) {
        SerError::OutOfDomain {
            path: classify_path_from_message(&msg),
            why: msg,
        }
    } else {
        SerError::Malformed {
            at: byte_offset,
            why: msg,
        }
    }
}

/// Compute an approximate byte offset from a 1-based `(line, column)` pair and
/// the raw input bytes.  Returns `input.len()` if the line/column is out of range
/// (e.g. for truncated inputs where serde_json reports line 1/col 1 on empty).
fn approx_byte_offset(input: &[u8], line: usize, col: usize) -> u64 {
    if input.is_empty() || line == 0 {
        return input.len() as u64;
    }
    let mut current_line = 1usize;
    let mut line_start = 0usize;
    for (i, &b) in input.iter().enumerate() {
        if current_line == line {
            // column is 1-based byte column within the line
            let col_offset = col.saturating_sub(1);
            return (line_start + col_offset).min(input.len()) as u64;
        }
        if b == b'\n' {
            current_line += 1;
            line_start = i + 1;
        }
    }
    // line > number of lines in input → truncated
    input.len() as u64
}

/// Same as [`map_serde_error`] but for string input (locus is a byte offset into
/// the UTF-8 string bytes).
fn map_serde_error_str(e: serde_json::Error, input: &str) -> SerError {
    map_serde_error(e, input.as_bytes())
}

/// Detect truncated/EOF errors: the input ended before a complete value was read.
fn is_truncated_error_msg(msg: &str, input: &[u8]) -> bool {
    // serde_json reports EOF errors with "EOF" or "unexpected end" in the message.
    // Also classify an empty input as truncated.
    let lower = msg.to_lowercase();
    input.is_empty() || lower.contains("eof") || lower.contains("unexpected end")
}

/// Detect domain errors: a field decoded successfully but violates a value-model
/// invariant (reported by `Value::new` → `serde::de::Error::custom`).
fn is_domain_error(msg: &str) -> bool {
    // Invariant-violation messages from `WfError::Display` and `Value::new`.
    msg.contains("payload")
        || msg.contains("repr")
        || msg.contains("guarantee")
        || msg.contains("bound")
        || msg.contains("invariant")
        || msg.contains("well-formed")
        || msg.contains("width")
}

/// Detect unknown-tag errors: an unrecognized `Repr`/ctor/`Meta` discriminant.
fn is_unknown_tag_error(msg: &str) -> bool {
    msg.contains("unknown variant")
        || msg.contains("unknown field")
        || msg.contains("expected one of")
}

/// Extract the unknown tag string from a `serde_json` "unknown variant X" message.
fn extract_unknown_tag(msg: &str) -> String {
    // Typical form: "unknown variant `Foo`, expected one of …"
    if let Some(start) = msg.find('`') {
        if let Some(end) = msg[start + 1..].find('`') {
            return msg[start + 1..start + 1 + end].to_owned();
        }
    }
    msg.to_owned()
}

/// Infer a field path from the error message (best-effort; see C3).
fn classify_path_from_message(msg: &str) -> FieldPath {
    if msg.contains("payload") {
        FieldPath::from_static("payload")
    } else if msg.contains("bound") {
        FieldPath::from_static("meta/bound")
    } else if msg.contains("repr") || msg.contains("width") {
        FieldPath::from_static("repr")
    } else {
        FieldPath::from_static("<unknown>")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// White-box unit tests live in a dedicated in-crate module (`src/tests/serialize.rs`),
// declared from `lib.rs` as `mod unit_tests`, per the "no tests in logic files" house
// rule (M-797 as-touched) — not inline here.
