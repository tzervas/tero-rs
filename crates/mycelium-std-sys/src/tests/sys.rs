//! White-box tests for [`crate::sys`] — env reads and the never-silent `args` parse (DN-40 §3).
//!
//! The non-UTF-8 cases exercise the [`parse_args`](crate::sys::parse_args) core mapping over a
//! crafted iterator of [`OsString`](std::ffi::OsString), so the never-silent property is pinned
//! without depending on the real process `argv` (which is all-UTF-8 in any sane test runner).

use crate::sys::*;
use std::ffi::OsString;

/// Build an `OsString` containing the given raw bytes. On Unix an arbitrary byte sequence (including
/// invalid UTF-8) is a valid `OsString`; on non-Unix the bytes are interpreted leniently. The
/// non-UTF-8 cases below are `#[cfg(unix)]`-gated so they only run where invalid bytes are faithfully
/// representable.
#[cfg(unix)]
fn os_from_bytes(bytes: &[u8]) -> OsString {
    use std::os::unix::ffi::OsStringExt;
    OsString::from_vec(bytes.to_vec())
}

/// Never-silent (G2): an unset variable is an explicit `None`, not an empty string. Uses a name
/// that is overwhelmingly unlikely to be set in any environment.
#[test]
fn an_unset_env_var_is_none_not_empty_string() {
    let name = "MYCELIUM_STD_SYS_DEFINITELY_UNSET_VARIABLE_X9Z";
    assert_eq!(get_env(name), None, "an unset var must be None, never \"\"");
}

/// An existing variable round-trips through the reader verbatim (the OS env is ground truth).
/// Reads a pre-existing var instead of mutating the global environment — `std::env::set_var`
/// is not thread-safe with the concurrent env reads other tests perform (Copilot #507). Skips
/// silently when the process starts with an empty environment (`env -i` / hermetic runners): the
/// unset-var test above already covers the empty-env case, so this must not panic on it (#508).
#[test]
fn an_existing_env_var_reads_back_verbatim() {
    if let Some((key, value)) = std::env::vars().next() {
        assert_eq!(get_env(&key), Some(value));
    }
}

/// `args()` over the real `argv` always contains at least the program name (arg 0). The real argv is
/// UTF-8 in any standard runner, so this is `Ok` and non-empty.
#[test]
fn args_includes_arg_zero() {
    let parsed = args().expect("a standard test runner's argv is valid UTF-8");
    assert!(!parsed.is_empty(), "args must include arg 0");
}

/// Valid args pass through `parse_args` verbatim and in order — the happy path is lossless.
#[test]
fn valid_args_pass_through_in_order() {
    let raw = vec![
        OsString::from("prog"),
        OsString::from("--flag"),
        OsString::from("value"),
    ];
    let parsed = parse_args(raw).expect("all-UTF-8 input parses");
    assert_eq!(parsed, vec!["prog", "--flag", "value"]);
}

/// `parse_args` over an empty iterator is an empty `Vec` — total over no input, never an error.
#[test]
fn empty_argv_is_ok_empty() {
    let parsed = parse_args(Vec::<OsString>::new()).expect("empty argv parses");
    assert!(parsed.is_empty());
}

/// Never-silent (G2): a non-UTF-8 argument is an `Err` naming its **index**, NOT silently dropped.
/// The previous `filter_map` would have returned `["prog", "after"]` — vanishing the bad arg and
/// shifting `after` from index 2 to index 1. We assert the explicit error instead.
#[cfg(unix)]
#[test]
fn a_non_utf8_arg_is_an_error_naming_its_index_not_a_silent_drop() {
    // 0xFF is never a valid UTF-8 byte.
    let raw = vec![
        OsString::from("prog"),
        os_from_bytes(&[b'b', b'a', b'd', 0xFF]),
        OsString::from("after"),
    ];
    let err = parse_args(raw).expect_err("a non-UTF-8 arg must surface an error, never be dropped");
    assert_eq!(
        err.index, 1,
        "the error must name the offending arg's index"
    );
    // The lossy rendering shows the arg (with U+FFFD for the bad byte) — diagnostic, never fabricated
    // as a faithful round-trip.
    assert!(
        err.lossy.starts_with("bad"),
        "lossy rendering should carry the recoverable prefix, got {:?}",
        err.lossy
    );
}

/// The error short-circuits on the **first** non-UTF-8 arg and reports that arg's index (not a later
/// one), so the offending position is unambiguous.
#[cfg(unix)]
#[test]
fn the_error_reports_the_first_non_utf8_arg() {
    let raw = vec![
        OsString::from("prog"),
        OsString::from("ok"),
        os_from_bytes(&[0xFF]),
        os_from_bytes(&[0xFE]),
    ];
    let err = parse_args(raw).expect_err("a non-UTF-8 arg must error");
    assert_eq!(err.index, 2, "must report the first offending index");
}

/// The error `Display` names the index and the lossy rendering — an inspectable, EXPLAIN-able message
/// (G2/no-black-boxes), not an opaque failure.
#[cfg(unix)]
#[test]
fn the_error_display_is_inspectable() {
    let raw = vec![OsString::from("prog"), os_from_bytes(&[0xFF])];
    let err = parse_args(raw).expect_err("a non-UTF-8 arg must error");
    let msg = err.to_string();
    assert!(
        msg.contains("index 1"),
        "message must name the index: {msg}"
    );
    assert!(
        msg.contains("not valid UTF-8"),
        "message must state the cause: {msg}"
    );
}
