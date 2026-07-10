//! Shared harness for the Stage-5 self-hosting **marshalling differential** (M-1013 STEP 2/3;
//! DN-26 §10.2). The generic, type-agnostic primitives every self-hosted `compiler.semcore` port
//! increment reuses: the `semcore.myc` include + driver `program`, the never-silent `L1Value`
//! decoders for the polymorphic wrappers (`Vec`/`Option`/`Result`) and the leaf reprs, the
//! `assert_l1_marshal`/`decode_driver` runners, and the surface-INPUT encoders that build a driver's
//! argument text. Type-specific decoders/encoders (a module's own ADT mirrors) stay in that
//! increment's test file; only the shared machinery lives here (extracted from
//! `compiler_stage5_elab.rs` at STEP 3, behaviour-preserving).
//!
//! The method: run the REAL Rust oracle on a fixture → evaluate the `.myc` port's mirror driver →
//! DECODE the returned `L1Value` back into the real Rust type (the `decode_*` family, the never-silent
//! inverse of the mirror constructors) → compare with Rust's own trusted derived `==` (`assert_eq!`).
//! A mis-lowering diverges the decoded value from the oracle; a decoder that drops a dimension is
//! caught by each increment's `marshal_discriminates` non-vacuity twin.

use crate::ast::{BaseType, Scalar, Sparsity, TypeRef, WidthRef};
use crate::checkty::{check_nodule, Ty, Width};
use crate::eval::{Evaluator, L1Value};
use crate::mono::monomorphize;
use crate::parse;
use mycelium_core::{Payload, Value};

pub(crate) const SEMCORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/semcore.myc"
));

/// Append a `main`-carrying driver to the self-hosted frontend, forming a runnable nodule.
pub(crate) fn program(driver: &str) -> String {
    format!("{SEMCORE_SRC}\n{driver}")
}

// ── L1Value → mycelium_core / checked-type decoders (the marshalling inverse) ───────────────────────
//
// Each decoder is the never-silent inverse of a mirror constructor: it walks the port's `L1Value`
// output and rebuilds the REAL Rust type, panicking (the harness's established failure mode) on an
// unexpected constructor rather than silently mis-decoding (G2). A mirror ADT node comes back as
// `L1Value::Data { ty, ctor, fields }`; a `Binary{N}`/`Bytes` leaf comes back as `L1Value::Repr(Value)`.

/// The base constructor name, with `monomorphize`'s injective mangle suffix stripped (`mono.rs` §4:
/// names are joined with `$`, with a `#` kind-tag on nullary data). A generic ctor specializes to
/// `Cons$Binary1`/`Some$Repr`/`Ok$Repr$Bytes`/…; the monomorphic mirror ctors (`CI`, `TyBinary`,
/// `RBinary`, …) carry no separator and pass through unchanged.
pub(crate) fn base_ctor(ctor: &str) -> &str {
    let end = ctor.find(['$', '#']).unwrap_or(ctor.len());
    &ctor[..end]
}

/// Every mirror ADT node is an `L1Value::Data`; return its (unmangled) constructor name + fields.
pub(crate) fn expect_data<'a>(v: &'a L1Value, what: &str) -> (&'a str, &'a [L1Value]) {
    match v {
        L1Value::Data { ctor, fields, .. } => (base_ctor(ctor), fields.as_slice()),
        other => panic!("marshal {what}: expected a Data node, got {other:?}"),
    }
}

/// A `Binary{32}` mirror int → `u32`, MSB-first — the exact convention `core_bits_as_u32` used.
pub(crate) fn decode_u32(v: &L1Value) -> u32 {
    match v.as_repr().map(Value::payload) {
        Some(Payload::Bits(bits)) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("marshal decode_u32: expected a Repr(Binary) Bits leaf, got {other:?}"),
    }
}

/// A `Binary{1}` mirror bit → `bool`.
pub(crate) fn decode_bit(v: &L1Value) -> bool {
    match v.as_repr().map(Value::payload) {
        Some(Payload::Bits(bits)) if bits.len() == 1 => bits[0],
        other => panic!("marshal decode_bit: expected a 1-bit Bits leaf, got {other:?}"),
    }
}

/// A `Bytes` leaf → raw bytes.
pub(crate) fn decode_bytes(v: &L1Value) -> Vec<u8> {
    match v.as_repr().map(Value::payload) {
        Some(Payload::Bytes(b)) => b.clone(),
        other => panic!("marshal decode_bytes: expected a Bytes leaf, got {other:?}"),
    }
}

/// A `Bytes` leaf → `String` (the fixtures are ASCII).
pub(crate) fn decode_string(v: &L1Value) -> String {
    String::from_utf8(decode_bytes(v)).expect("marshal decode_string: non-UTF8 bytes")
}

/// A `.myc` `Vec[A]` (`Nil | Cons(A, Vec[A])`) → `Vec<T>`, decoding each element with `elem`.
pub(crate) fn decode_vec<T>(v: &L1Value, elem: impl Fn(&L1Value) -> T) -> Vec<T> {
    let mut out = Vec::new();
    let mut cur = v;
    loop {
        let (ctor, fields) = expect_data(cur, "Vec");
        match ctor {
            "Nil" => return out,
            "Cons" => {
                out.push(elem(&fields[0]));
                cur = &fields[1];
            }
            other => panic!("marshal decode_vec: unexpected ctor {other}"),
        }
    }
}

/// A `.myc` `Option[A]` (`None | Some(A)`) → `Option<T>`.
pub(crate) fn decode_option<T>(v: &L1Value, elem: impl Fn(&L1Value) -> T) -> Option<T> {
    let (ctor, fields) = expect_data(v, "Option");
    match ctor {
        "None" => None,
        "Some" => Some(elem(&fields[0])),
        c => panic!("marshal decode_option: unexpected ctor {c}"),
    }
}

/// A `.myc` `Result[A, Bytes]` → `Result<T, ()>`. The `Err` arm's message differs across the two
/// independent productions (a Rust error vs the `.myc` `Bytes` string), so it is normalized to `()`
/// (any `Err` == any `Err`; only the `Ok` payload is a meaningful differential).
pub(crate) fn decode_result<T>(v: &L1Value, elem: impl Fn(&L1Value) -> T) -> Result<T, ()> {
    let (ctor, fields) = expect_data(v, "Result");
    match ctor {
        "Ok" => Ok(elem(&fields[0])),
        "Err" => Err(()),
        c => panic!("marshal decode_result: unexpected ctor {c}"),
    }
}

// ── the marshalling runners ─────────────────────────────────────────────────────────────────────────

/// Parse → check → monomorphize → eval `main` → DECODE the mirror `L1Value` → `assert_eq!` against the
/// trusted Rust oracle value. The comparator is Rust's own derived `==` (no `.myc`-side `_eq`).
pub(crate) fn assert_l1_marshal<T: PartialEq + std::fmt::Debug>(
    label: &str,
    driver: &str,
    decode: impl Fn(&L1Value) -> T,
    want: T,
) {
    let src = program(driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let got = decode(&l1_val);
    assert_eq!(
        got, want,
        "{label}: decoded marshal {got:?} does not match oracle {want:?}"
    );
}

/// Eval a bare mirror-literal driver and decode it — the `marshal_discriminates` primitive (no oracle;
/// used only to prove the decoder distinguishes single-dimension-distinct mirror values).
pub(crate) fn decode_driver<T>(ret_ty: &str, expr: &str, decode: impl Fn(&L1Value) -> T) -> T {
    let driver = format!("fn main() => {ret_ty} = {expr};\n");
    let src = program(&driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("decode_driver parse: {e}")))
        .unwrap_or_else(|e| panic!("decode_driver check: {e}"));
    let mono = monomorphize(&env, "main").unwrap_or_else(|e| panic!("decode_driver mono: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("decode_driver eval: {e}"));
    decode(&l1_val)
}

// ── Rust → `.myc` fixture encoders (surface INPUT types — build the driver's argument text) ─────────

pub(crate) fn encode_u32(n: u32) -> String {
    let mut s = String::from("0b");
    for (count, i) in (0..32).rev().enumerate() {
        if count != 0 && count % 4 == 0 {
            s.push('_');
        }
        s.push(if (n >> i) & 1 == 1 { '1' } else { '0' });
    }
    s
}

pub(crate) fn encode_bytes(s: &str) -> String {
    format!("{s:?}")
}

pub(crate) fn encode_scalar(s: Scalar) -> &'static str {
    match s {
        Scalar::F16 => "SF16",
        Scalar::Bf16 => "SBf16",
        Scalar::F32 => "SF32",
        Scalar::F64 => "SF64",
    }
}

pub(crate) fn encode_sparsity(sp: &Sparsity) -> String {
    match sp {
        Sparsity::Dense => "SpDense".to_owned(),
        Sparsity::Sparse(k) => format!("SpSparse({})", encode_u32(*k)),
    }
}

pub(crate) fn encode_width(w: &Width) -> String {
    match w {
        Width::Lit(n) => format!("WdLit({})", encode_u32(*n)),
        Width::Var(v) => format!("WdVar({})", encode_bytes(v)),
    }
}

pub(crate) fn encode_widthref(w: &WidthRef) -> String {
    match w {
        WidthRef::Lit(n) => format!("WLit({})", encode_u32(*n)),
        WidthRef::Name(v) => format!("WName({})", encode_bytes(v)),
    }
}

pub(crate) fn encode_ty(t: &Ty) -> String {
    match t {
        Ty::Binary(w) => format!("TyBinary({})", encode_width(w)),
        Ty::Ternary(w) => format!("TyTernary({})", encode_width(w)),
        Ty::Dense(d, s) => format!("TyDense({}, {})", encode_u32(*d), encode_scalar(*s)),
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => format!(
            "TyVsa({}, {}, {})",
            encode_bytes(model),
            encode_u32(*dim),
            encode_sparsity(sparsity)
        ),
        Ty::Data(n, args) => format!("TyData({}, {})", encode_bytes(n), encode_ty_list(args)),
        Ty::Substrate(t) => format!("TySubstrate({})", encode_bytes(t)),
        Ty::Seq(elem, n) => format!("TySeq({}, {})", encode_ty(elem), encode_u32(*n)),
        Ty::Bytes => "TyBytes".to_owned(),
        Ty::Float => "TyFloat".to_owned(),
        Ty::Var(v) => format!("TyVar({})", encode_bytes(v)),
        Ty::Fn(a, r) => format!("TyFn({}, {})", encode_ty(a), encode_ty(r)),
    }
}

pub(crate) fn encode_ty_list(ts: &[Ty]) -> String {
    let mut s = String::from("Nil");
    for t in ts.iter().rev() {
        s = format!("Cons({}, {})", encode_ty(t), s);
    }
    s
}

pub(crate) fn encode_typeref(t: &TypeRef) -> String {
    // The surface guarantee slot is discarded by every `resolve_ty` consumer exercised here
    // (`type_repr` ignores it; `resolve_ctors` does `let (ty, _) = ..`), so always emit `None`.
    format!("TR({}, None)", encode_basetype(&t.base))
}

pub(crate) fn encode_typeref_list(ts: &[TypeRef]) -> String {
    let mut s = String::from("Nil");
    for t in ts.iter().rev() {
        s = format!("Cons({}, {})", encode_typeref(t), s);
    }
    s
}

pub(crate) fn encode_basetype(b: &BaseType) -> String {
    match b {
        BaseType::Binary(w) => format!("KwBinary({})", encode_widthref(w)),
        BaseType::Ternary(w) => format!("KwTernary({})", encode_widthref(w)),
        BaseType::Dense(d, s) => format!("KwDense({}, {})", encode_u32(*d), encode_scalar(*s)),
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => format!(
            "Vsa({}, {}, {})",
            encode_bytes(model),
            encode_u32(*dim),
            encode_sparsity(sparsity)
        ),
        BaseType::Substrate(t) => format!("KwSubstrate({})", encode_bytes(t)),
        BaseType::Seq { elem, len } => {
            format!("KwSeq({}, {})", encode_typeref(elem), encode_u32(*len))
        }
        BaseType::Bytes => "KwBytes".to_owned(),
        BaseType::Float => "KwFloat".to_owned(),
        BaseType::Named(name, args) => {
            format!(
                "Named({}, {})",
                encode_bytes(name),
                encode_typeref_list(args)
            )
        }
        BaseType::Fn(a, r) => format!("FnArrow({}, {})", encode_typeref(a), encode_typeref(r)),
        BaseType::Tuple(elems) => format!("Tuple({})", encode_typeref_list(elems)),
        BaseType::Ambient(_) => {
            panic!("Ambient BaseType is not exercised by the marshalling differential")
        }
    }
}

/// A surface `TypeRef` with no guarantee slot — the common fixture constructor.
pub(crate) fn tref(base: BaseType) -> TypeRef {
    TypeRef {
        base,
        guarantee: None,
    }
}
