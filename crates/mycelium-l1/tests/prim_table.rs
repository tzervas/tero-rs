//! R7-Q4 (M-390) equivalence guard: the L1 surface prim table (`checkty::prim_sig` +
//! `prim_kernel_name`) must stay consistent with the content-addressed prim table `Π`
//! (`mycelium_core::PrimTable`). DN-10 §3.4: `Π_new(hash(p)) = Π_old(name(p))`.
//!
//! The surface table is L1 sugar: width-polymorphic surface names (`add`, `xor`, …) that elaborate
//! onto the trusted kernel prims (`trit.add`, `bit.xor`, …). This guard pins three properties so the
//! two hard-coded lists cannot drift: (1) every surface prim's kernel target is a *declared* kernel
//! prim in the table; (2) the table entry's arity matches what `prim_sig` accepts; (3) the operand
//! and result *paradigms* agree.

use mycelium_core::{PrimParadigm, PrimTable};
use mycelium_l1::checkty::{prim_kernel_name, prim_sig};
use mycelium_l1::{checkty::Width, Ty};

/// The paradigm a surface `Ty` operand presents to the kernel.
fn paradigm_of(t: &Ty) -> PrimParadigm {
    match t {
        Ty::Binary(_) => PrimParadigm::Binary,
        Ty::Ternary(_) => PrimParadigm::Ternary,
        other => panic!("prim operands are Binary/Ternary in v0, got {other:?}"),
    }
}

/// Each surface prim, with representative well-typed operands and the result `prim_sig` must yield.
fn surface_cases() -> Vec<(&'static str, Vec<Ty>, Ty)> {
    vec![
        (
            "not",
            vec![Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // CU-6: bit-manipulation counts (popcount/clz/ctz) — unary Binary{N} -> Binary{N}.
        (
            "popcount",
            vec![Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "clz",
            vec![Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "ctz",
            vec![Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "xor",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "add",
            vec![Ty::Ternary(Width::Lit(4)), Ty::Ternary(Width::Lit(4))],
            Ty::Ternary(Width::Lit(4)),
        ),
        (
            "sub",
            vec![Ty::Ternary(Width::Lit(4)), Ty::Ternary(Width::Lit(4))],
            Ty::Ternary(Width::Lit(4)),
        ),
        (
            "mul",
            vec![Ty::Ternary(Width::Lit(4)), Ty::Ternary(Width::Lit(4))],
            Ty::Ternary(Width::Lit(4)),
        ),
        (
            "neg",
            vec![Ty::Ternary(Width::Lit(4))],
            Ty::Ternary(Width::Lit(4)),
        ),
        // RFC-0032 D2 (M-748): width-uniform binary logical + never-silent arithmetic.
        (
            "and",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "or",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "add_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "sub_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement multiply.
        (
            "mul_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2 (CU-1): never-silent UNSIGNED multiply (`bit.mul`) — overflow-distinct
        // from the signed `mul_s`/`bin.mul`; the `math.myc` FLAG-math-1 missing op.
        (
            "mul_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent unsigned division/remainder.
        (
            "div_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "rem_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent logical left/right shift.
        (
            "shl_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "shr_u",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2/§4.1.3 (M-766, `enb` Gap B): never-silent two's-complement add/sub/neg —
        // completes the shared set `mul_s` started. The `_s` suffix marks the signed reading,
        // distinct from the unsigned `add_u`/`sub_u` (`bit.add`/`bit.sub`) — DN-72.
        (
            "add_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "sub_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "neg_s",
            vec![Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        // RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the signedness-split signed op set —
        // signed truncated division/remainder + the arithmetic right shift, distinct-named from
        // their `_u` counterparts (ADR-028; DN-72). (`lt_s` is width-collapsing and rides the
        // comparison guard below, not this width-uniform table.)
        (
            "div_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "rem_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
        (
            "shr_s",
            vec![Ty::Binary(Width::Lit(8)), Ty::Binary(Width::Lit(8))],
            Ty::Binary(Width::Lit(8)),
        ),
    ]
}

/// The M-890 (`enb` Gap C) dense elementwise prims are **tensor-valued** — their operands/results
/// are `Ty::Dense(dim, scalar)`, typed by a dedicated checker branch (`try_check_dense_prim`), and
/// their Π entries use the documented `Any`/`Uniform` paradigm-model escape hatch (no first-class
/// `Dense` paradigm yet — the same FLAG as the seq/bytes prims). So they fit neither
/// `surface_cases` (which asserts Binary/Ternary operand paradigms) nor the comparison guard. This
/// guard pins their surface→Π consistency directly: each surface name maps to a declared kernel
/// prim with the right arity, `Any` operands/result, and — the M-890 core contract — the intrinsic
/// tag **carried from the kernel** (`dense.neg` `Exact`; `dense.add`/`dense.sub`/`dense.scale`
/// `Proven`; VR-5 — the kernel-side twin lives in `mycelium-interp/tests/prim_table.rs`, which can
/// see `DenseSpace::op_guarantee` itself).
#[test]
fn dense_prims_resolve_to_declared_tensor_valued_kernel_prims() {
    use mycelium_core::GuaranteeStrength;
    let table = PrimTable::builtins();
    for (surface, kernel_expected, arity, intrinsic) in [
        ("dense_add", "dense.add", 2, GuaranteeStrength::Proven),
        ("dense_sub", "dense.sub", 2, GuaranteeStrength::Proven),
        ("dense_neg", "dense.neg", 1, GuaranteeStrength::Exact),
        ("dense_scale", "dense.scale", 2, GuaranteeStrength::Proven),
        // M-891: the measurement pair — Proven (the kernel's binary64 accumulation bound;
        // the runtime result is Dense{1, F64}, typed by the dedicated checker branch).
        ("dense_dot", "dense.dot", 2, GuaranteeStrength::Proven),
        (
            "dense_similarity",
            "dense.similarity",
            2,
            GuaranteeStrength::Proven,
        ),
    ] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("dense prim `{surface}` must map to a kernel name"));
        assert_eq!(kernel, kernel_expected, "surface→kernel mapping drifted");
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), arity, "`{kernel}` arity drifted");
        assert!(
            decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "`{kernel}` operands use the documented `Any` escape hatch (no Dense paradigm yet)",
        );
        assert_eq!(decl.sig.result, PrimParadigm::Any);
        assert_eq!(
            decl.intrinsic, intrinsic,
            "`{kernel}` intrinsic must be carried from the kernel's op_guarantee (VR-5)",
        );
    }
}

/// The M-898 (`enb` Gap A) scalar-float arithmetic prims operate on the nullary `Ty::Float`
/// (IEEE-754 binary64 — ADR-040), typed by a dedicated checker branch (`try_check_float_prim`),
/// and their Π entries use the documented `Any`/`Uniform` paradigm-model escape hatch (no
/// first-class `Float` paradigm yet — the same FLAG as the seq/bytes/dense prims). This guard
/// pins their surface→Π consistency directly: each `flt_*` surface name maps to a declared
/// `flt.*` kernel prim with the right arity, `Any` operands/result, and — the M-898 core
/// contract — the intrinsic at the ratified ADR-040 §2.6 **`Empirical`** (VR-5: the
/// host-conformance claim, never upgraded; the value-side twin — tag + zero-deviation bound —
/// lives in `mycelium-interp`).
#[test]
fn float_prims_resolve_to_declared_empirical_kernel_prims() {
    use mycelium_core::GuaranteeStrength;
    let table = PrimTable::builtins();
    for (surface, kernel_expected, arity) in [
        ("flt_add", "flt.add", 2),
        ("flt_sub", "flt.sub", 2),
        ("flt_mul", "flt.mul", 2),
        ("flt_div", "flt.div", 2),
        ("flt_neg", "flt.neg", 1),
    ] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("float prim `{surface}` must map to a kernel name"));
        assert_eq!(kernel, kernel_expected, "surface→kernel mapping drifted");
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), arity, "`{kernel}` arity drifted");
        assert!(
            decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "`{kernel}` operands use the documented `Any` escape hatch (no Float paradigm yet)",
        );
        assert_eq!(decl.sig.result, PrimParadigm::Any);
        assert_eq!(
            decl.intrinsic,
            GuaranteeStrength::Empirical,
            "`{kernel}` intrinsic must stay the ratified ADR-040 §2.6 Empirical (VR-5)",
        );
    }
}

/// ADR-040 §2.4 (CU-3): the never-silent Binary↔Float conversions — mixed-paradigm operands
/// (unlike the uniform-Float arithmetic group above), typed by a dedicated pre-branch in
/// `try_check_float_prim` rather than the uniform loop. This guard pins their surface→Π
/// consistency: each `bin_to_flt`/`flt_to_bin` surface name maps to a declared kernel prim with
/// the right arity, the documented `Any` operand/result escape hatch (no first-class `Float`
/// paradigm yet), and the intrinsic at the ratified ADR-040 §2.6 **`Empirical`** ("Conversions:
/// range/exactness checks Empirical…" — never `Exact`, VR-5).
#[test]
fn float_conversion_prims_resolve_to_declared_empirical_kernel_prims() {
    use mycelium_core::GuaranteeStrength;
    let table = PrimTable::builtins();
    for (surface, kernel_expected, arity) in [
        ("bin_to_flt", "bin.to_flt", 1),
        ("flt_to_bin", "flt.to_bin", 2),
    ] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("conversion prim `{surface}` must map to a kernel name"));
        assert_eq!(kernel, kernel_expected, "surface→kernel mapping drifted");
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), arity, "`{kernel}` arity drifted");
        assert!(
            decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "`{kernel}` operands use the documented `Any` escape hatch (no Float paradigm yet)",
        );
        assert_eq!(decl.sig.result, PrimParadigm::Any);
        assert_eq!(
            decl.intrinsic,
            GuaranteeStrength::Empirical,
            "`{kernel}` intrinsic must be the ratified ADR-040 §2.6 Empirical — conversions are \
             never `Exact` (VR-5)",
        );
    }
}

/// The M-899 (`enb` Gap A) scalar-float comparison prims — the IEEE-754 §5.11 partial-order
/// predicates (`flt_lt`/`flt_le`/`flt_gt`/`flt_ge`/`flt_eq`, NaN explicitly unordered → false)
/// plus the named opt-in total order `flt_total_le` (IEEE-754 §5.10 `totalOrder`) — are
/// **width-collapsing** like the D1 pair: two `Float` operands reduce to a `Binary{1}` truth
/// value (typed by the `try_check_float_prim` branch). This guard pins their surface→Π
/// consistency: each `flt_*` comparison maps to a declared collapsing `flt.*` kernel prim with
/// arity 2, `Any` operands (the documented no-Float-paradigm escape hatch), a genuinely
/// `Binary` result, and the intrinsic at the ratified ADR-040 §2.6 **`Empirical`** — for
/// `flt.total_le` that tag is load-bearing: the total-order property is the **M-511 proof
/// debt**, `Empirical` until a proof lands, never `Proven` on host documentation (VR-5).
#[test]
fn float_cmp_prims_resolve_to_declared_collapsing_empirical_kernel_prims() {
    use mycelium_core::{GuaranteeStrength, WidthRel};
    let table = PrimTable::builtins();
    for (surface, kernel_expected) in [
        ("flt_lt", "flt.lt"),
        ("flt_le", "flt.le"),
        ("flt_gt", "flt.gt"),
        ("flt_ge", "flt.ge"),
        ("flt_eq", "flt.eq"),
        ("flt_total_le", "flt.total_le"),
    ] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("float cmp prim `{surface}` must map to a kernel name"));
        assert_eq!(kernel, kernel_expected, "surface→kernel mapping drifted");
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), 2, "`{kernel}` is binary (two operands)");
        assert!(
            decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "`{kernel}` operands use the documented `Any` escape hatch (no Float paradigm yet)",
        );
        assert_eq!(
            decl.sig.result,
            PrimParadigm::Binary,
            "`{kernel}` reduces to a Binary{{1}} truth value",
        );
        assert_eq!(
            decl.sig.width,
            WidthRel::Collapse,
            "`{kernel}` is width-collapsing (two Float scalars → Binary{{1}})",
        );
        assert_eq!(
            decl.intrinsic,
            GuaranteeStrength::Empirical,
            "`{kernel}` intrinsic must stay the ratified ADR-040 §2.6 Empirical (VR-5; the \
             flt.total_le total-order property is the M-511 proof debt — never `Proven` \
             without the checked theorem)",
        );
    }
}

/// The RFC-0032 D1 (M-747) comparison prims are **width-collapsing** and paradigm-flexible
/// (`Any, Any → Binary`, `WidthRel::Collapse`), so they do not fit the width-uniform `surface_cases`
/// shape (they bypass `prim_sig` via a dedicated checker branch). This guard pins their surface→Π
/// consistency directly: each maps to a declared collapsing kernel prim with arity 2.
#[test]
fn comparison_prims_resolve_to_declared_collapsing_kernel_prims() {
    use mycelium_core::WidthRel;
    let table = PrimTable::builtins();
    // `lt_s` (M-767) is the signed order — same collapsing shape as the D1 pair (its Π operands
    // are pinned `Binary` rather than the D1 `Any`, checked by `mycelium-core`'s own Π test).
    for surface in ["eq", "lt", "lt_s"] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("comparison prim `{surface}` must map to a kernel name"));
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), 2, "`{kernel}` is binary (two operands)");
        assert_eq!(
            decl.sig.result,
            PrimParadigm::Binary,
            "`{kernel}` reduces to a Binary truth value",
        );
        assert_eq!(
            decl.sig.width,
            WidthRel::Collapse,
            "`{kernel}` is width-collapsing (operands' width → Binary{{1}})",
        );
    }
}

#[test]
fn surface_prims_resolve_to_declared_kernel_prims() {
    let table = PrimTable::builtins();
    for (surface, args, _ret) in surface_cases() {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("surface prim `{surface}` must map to a kernel name"));
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not a declared prim in Π",
        );
        let _ = args;
    }
}

#[test]
fn surface_signature_matches_the_kernel_declaration() {
    let table = PrimTable::builtins();
    for (surface, args, ret) in surface_cases() {
        // `prim_sig` accepts the representative operands and yields the expected result type.
        assert_eq!(
            prim_sig(surface, &args),
            Some(ret.clone()),
            "surface `{surface}` signature changed unexpectedly",
        );

        let kernel = prim_kernel_name(surface).expect("kernel name");
        let decl = table.get(kernel).expect("declared prim");

        // Arity agrees.
        assert_eq!(
            decl.sig.arity(),
            args.len(),
            "surface `{surface}` arity disagrees with kernel `{kernel}` declaration",
        );
        // Per-operand paradigm agrees.
        for (i, arg) in args.iter().enumerate() {
            assert_eq!(
                decl.sig.operands[i],
                paradigm_of(arg),
                "surface `{surface}` operand {i} paradigm disagrees with kernel `{kernel}`",
            );
        }
        // Result paradigm agrees.
        assert_eq!(
            decl.sig.result,
            paradigm_of(&ret),
            "surface `{surface}` result paradigm disagrees with kernel `{kernel}`",
        );
    }
}

/// The M-892 (`enb` Gap C) VSA bind-group prims are **model-dispatched** hypervector ops — their
/// operands/results are `Ty::Vsa{model, dim, sparsity}`, typed by a dedicated checker branch
/// (`try_check_vsa_prim`, which also gates the MAP-I/FHRR/BSC dispatch set statically), and their
/// Π entries use the documented `Any`/`Uniform` paradigm-model escape hatch (no first-class `Vsa`
/// paradigm yet — the same FLAG as the seq/bytes/dense/flt prims). This guard pins their
/// surface→Π consistency directly: each `vsa_*` surface name maps to a declared `vsa.*` kernel
/// prim with the right arity, `Any` operands/result, and — the M-892 core contract — the
/// intrinsic at the **meet over the dispatch set** (`vsa.bind`/`vsa.permute` `Exact`;
/// `vsa.unbind` `Empirical` — FHRR's normative weak-link unbind holds the meet down; VR-5: one Π
/// slot must not over-claim for any model. The kernel-side twin lives in
/// `mycelium-interp/tests/prim_table.rs`, which recomputes the meet from `mycelium-vsa` itself;
/// the runtime value carries the dispatched model's own tag). The M-893 certified superposition
/// `vsa_bundle` rides the same shape (`Seq{VSA{…}, N}` × `Float` δ under the `Any` hatch) with
/// its intrinsic the meet over its **certified singleton** dispatch set {MAP-I} = `Proven` (the
/// checked `CapacityBound` rides the runtime value; an FHRR/BSC bundle is a *static* refusal in
/// `try_check_vsa_prim` naming the certified set). The M-894 trio rides the same shape:
/// `vsa_cleanup`/`vsa_reconstruct` are `Exact` — the RFC-0010 §4.4 exhaustive-arg-max decode
/// claim, met over their dispatch sets (MAP-I/FHRR/BSC and {MAP-I, BSC} respectively; the runtime
/// triple carries the query/record's own (strength, bound) pair) — and `vsa_required_dim` is
/// `Proven` (the M-131 checked instantiation; its result carries the kernel's `CapacityBound`).
#[test]
fn vsa_prims_resolve_to_declared_model_dispatched_kernel_prims() {
    use mycelium_core::GuaranteeStrength;
    let table = PrimTable::builtins();
    for (surface, kernel_expected, arity, intrinsic) in [
        ("vsa_bind", "vsa.bind", 2, GuaranteeStrength::Exact),
        ("vsa_unbind", "vsa.unbind", 2, GuaranteeStrength::Empirical),
        ("vsa_permute", "vsa.permute", 2, GuaranteeStrength::Exact),
        ("vsa_bundle", "vsa.bundle", 2, GuaranteeStrength::Proven),
        ("vsa_cleanup", "vsa.cleanup", 2, GuaranteeStrength::Exact),
        (
            "vsa_reconstruct",
            "vsa.reconstruct",
            4,
            GuaranteeStrength::Exact,
        ),
        (
            "vsa_required_dim",
            "vsa.required_dim",
            2,
            GuaranteeStrength::Proven,
        ),
    ] {
        let kernel = prim_kernel_name(surface)
            .unwrap_or_else(|| panic!("vsa prim `{surface}` must map to a kernel name"));
        assert_eq!(kernel, kernel_expected, "surface→kernel mapping drifted");
        assert!(
            table.contains(kernel),
            "surface `{surface}` → kernel `{kernel}`, but `{kernel}` is not declared in Π",
        );
        let decl = table.get(kernel).expect("declared prim");
        assert_eq!(decl.sig.arity(), arity, "`{kernel}` arity drifted");
        assert!(
            decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "`{kernel}` operands use the documented `Any` escape hatch (no Vsa paradigm yet)",
        );
        assert_eq!(decl.sig.result, PrimParadigm::Any);
        assert_eq!(
            decl.intrinsic, intrinsic,
            "`{kernel}` intrinsic must be the meet over its dispatch set / the checked \
             instantiation stance (MAP-I/FHRR/BSC for the bind group + cleanup; the certified \
             singleton {{MAP-I}} for vsa.bundle; {{MAP-I, BSC}} for vsa.reconstruct; the M-131 \
             ProvenThm stance for vsa.required_dim — VR-5)",
        );
    }
}

/// The M-912 (`enb`) additions — `bytes_eq` (the folded-in equality gap the diag/error/recover
/// ports flagged: `bytes.*` had len/get/slice/concat but no equality) and `hash_blake3` (the
/// kernel's own BLAKE3 content-addressing hash, M-103, surfaced as a prim). Both are typed by the
/// dedicated `try_check_seq_bytes_prim` branch (like the rest of the `bytes.*` group) and their Π
/// entries use the documented `Any` paradigm-model escape hatch (no first-class `Bytes` paradigm
/// yet). `bytes_eq` genuinely reduces to a `Binary` truth value (not a hatch); `hash_blake3`'s
/// result is `Any` like `bytes_slice`/`bytes_concat`. Both `Exact` — `bytes_eq` is a total `[u8]`
/// comparison, and `hash_blake3` is `Exact` on the strength of the kernel's own deterministic
/// BLAKE3 use (M-103), pinned by the known-digest conformance vectors in `mycelium-interp`.
#[test]
fn bytes_eq_and_hash_blake3_resolve_to_declared_kernel_prims() {
    use mycelium_core::GuaranteeStrength;
    let table = PrimTable::builtins();

    let bytes_eq_kernel = prim_kernel_name("bytes_eq").expect("bytes_eq must map to a kernel name");
    assert_eq!(
        bytes_eq_kernel, "bytes.eq",
        "surface→kernel mapping drifted"
    );
    assert!(
        table.contains(bytes_eq_kernel),
        "surface `bytes_eq` → kernel `{bytes_eq_kernel}`, but it is not declared in Π",
    );
    let decl = table.get(bytes_eq_kernel).expect("declared prim");
    assert_eq!(decl.sig.arity(), 2, "`bytes.eq` is binary (two operands)");
    assert!(
        decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
        "`bytes.eq` operands use the documented `Any` escape hatch (no Bytes paradigm yet)",
    );
    assert_eq!(
        decl.sig.result,
        PrimParadigm::Binary,
        "`bytes.eq` reduces to a Binary{{1}} truth value",
    );
    assert_eq!(decl.intrinsic, GuaranteeStrength::Exact);

    let hash_kernel =
        prim_kernel_name("hash_blake3").expect("hash_blake3 must map to a kernel name");
    assert_eq!(hash_kernel, "hash.blake3", "surface→kernel mapping drifted");
    assert!(
        table.contains(hash_kernel),
        "surface `hash_blake3` → kernel `{hash_kernel}`, but it is not declared in Π",
    );
    let decl = table.get(hash_kernel).expect("declared prim");
    assert_eq!(decl.sig.arity(), 1, "`hash.blake3` is unary");
    assert!(
        decl.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
        "`hash.blake3` operand uses the documented `Any` escape hatch (no Bytes paradigm yet)",
    );
    assert_eq!(decl.sig.result, PrimParadigm::Any);
    assert_eq!(
        decl.intrinsic,
        GuaranteeStrength::Exact,
        "`hash.blake3` intrinsic must stay Exact — justified by the kernel's own deterministic \
         BLAKE3 use (VR-5)",
    );
}
