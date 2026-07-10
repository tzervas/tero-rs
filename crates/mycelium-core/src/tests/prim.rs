//! White-box tests for [`crate::prim`]. Extracted from the logic file (test-layout rule, M-797).

use crate::guarantee::GuaranteeStrength;
use crate::prim::{PrimDecl, PrimParadigm, PrimSig, PrimTable, WidthRel};

fn xor() -> PrimDecl {
    PrimDecl {
        sig: PrimSig {
            operands: vec![PrimParadigm::Binary, PrimParadigm::Binary],
            result: PrimParadigm::Binary,
            width: WidthRel::Uniform,
        },
        intrinsic: GuaranteeStrength::Exact,
    }
}

#[test]
fn hash_is_well_shaped_blake3_and_name_independent() {
    let h = xor().content_hash();
    assert_eq!(h.algo(), "blake3");
    assert_eq!(h.digest().len(), 64);
    // The same declaration under two different kernel names has the same identity (ADR-003).
    let mut t = PrimTable::new();
    let a = t.insert("bit.xor", xor());
    let b = t.insert("bit.xor_alias", xor());
    assert_eq!(a, b, "identity is the signature+intrinsic, not the name");
}

#[test]
fn distinct_signatures_get_distinct_identities() {
    let not = PrimDecl {
        sig: PrimSig {
            operands: vec![PrimParadigm::Binary],
            result: PrimParadigm::Binary,
            width: WidthRel::Uniform,
        },
        intrinsic: GuaranteeStrength::Exact,
    };
    assert_ne!(
        xor().content_hash(),
        not.content_hash(),
        "different arity/paradigm ⇒ different identity"
    );
}

#[test]
fn intrinsic_is_identity_bearing() {
    // A prim whose only difference is the intrinsic guarantee is a *different* declaration —
    // the honesty tag is part of identity (so an Exact prim can never alias an Empirical one).
    let mut declared = xor();
    declared.intrinsic = GuaranteeStrength::Declared;
    assert_ne!(xor().content_hash(), declared.content_hash());
}

#[test]
fn builtins_are_present_and_resolvable() {
    let t = PrimTable::builtins();
    for name in [
        "core.id",
        "bit.not",
        "bit.and",
        "bit.or",
        "bit.xor",
        "trit.neg",
        "trit.add",
        "trit.sub",
        "trit.mul",
        "cmp.eq",
        "cmp.lt",
        "bit.add",
        "bit.sub",
        "bin.mul",
        "bin.div",
        "bin.rem",
        "bin.shl",
        "bin.shr",
        "bin.add",
        "bin.sub",
        "bin.neg",
        "bin.div_s",
        "bin.rem_s",
        "bin.shr_s",
        "cmp.lt_s",
        "bit.width_cast",
        "seq.len",
        "seq.get",
        "bytes.len",
        "bytes.get",
        "bytes.slice",
        "bytes.concat",
        "bytes.eq",
        "hash.blake3",
        "fuse_join:binary",
    ] {
        let r = t.prim_ref(name).expect("builtin registered");
        let d = t.resolve(&r).expect("ref resolves");
        assert_eq!(d.intrinsic, GuaranteeStrength::Exact);
        assert_eq!(t.intrinsic(name), Some(GuaranteeStrength::Exact));
    }
    // `entries()` is the EXPLAIN surface: one inspectable entry per builtin (RFC-0032 D1/D2
    // added cmp.eq/cmp.lt/bit.add/bit.sub to the original nine; D3/M-749 added seq.len/seq.get;
    // D4/M-750 added bytes.len/get/slice/concat; DN-41/M-798 added bit.width_cast; DN-58/M-817
    // added the `Binary` `Fuse` meet `fuse_join:binary`; RFC-0033/M-887 added `bin.mul`; RFC-0033/
    // M-888 added `bin.div`/`bin.rem`; RFC-0033/M-889 added `bin.shl`/`bin.shr`; RFC-0033/M-766
    // added `bin.add`/`bin.sub`/`bin.neg`, completing the shared two's-complement set; RFC-0001
    // §4.1/M-890 added the dense elementwise group `dense.add`/`dense.sub`/`dense.neg`/
    // `dense.scale`, the first tensor-valued prims, and M-891 the measurement pair
    // `dense.dot`/`dense.similarity` — pinned separately below because their
    // intrinsics are NOT all `Exact`; ADR-040 §2.5/M-898 added the scalar-float arithmetic
    // group `flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg` — likewise pinned separately
    // below, its intrinsic is `Empirical` per the ratified ADR-040 §2.6; ADR-040 §2.4/M-899
    // added the scalar-float comparison group `flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq`
    // (the IEEE-754 §5.11 partial-order predicates, NaN unordered) plus the named opt-in
    // total order `flt.total_le` — likewise `Empirical`, pinned separately below; the
    // total-order property stays unproven until the M-511 proof debt is discharged;
    // RFC-0033/M-767 added the signedness-split signed set `bin.div_s`/`bin.rem_s`/`bin.shr_s`
    // + the two's-complement ordering `cmp.lt_s`, the distinct-named signed counterparts to
    // `bin.div`/`bin.rem`/`bin.shr`/`cmp.lt` per ADR-028; RFC-0003 §3/§4/M-892 added the
    // model-dispatched VSA bind group `vsa.bind`/`vsa.unbind`/`vsa.permute` — pinned separately
    // below, `vsa.unbind`'s intrinsic is the `Empirical` meet over the MAP-I/FHRR/BSC dispatch
    // set; RFC-0003 §4/§5/M-893 added the certified superposition `vsa.bundle` — pinned
    // separately below, its intrinsic is `Proven` (the meet over its certified singleton
    // dispatch set {MAP-I}); RFC-0003 §3/§5/§6/M-894 added the cleanup/reconstruction pair
    // `vsa.cleanup`/`vsa.reconstruct` and the capacity query `vsa.required_dim` — pinned
    // separately below, `vsa.required_dim`'s intrinsic is `Proven` (the M-131 checked
    // instantiation); M-912 (`enb`) added `bytes.eq` (the folded-in equality gap) and
    // `hash.blake3` (the kernel's own BLAKE3 content-addressing hash surfaced as a prim) — both
    // `Exact`, listed above with the rest of the `Exact` group). CU-1 added `bit.mul` (never-silent
    // unsigned multiply, RFC-0033 §4.1.2 — the `math.myc` FLAG-math-1 missing op), `Exact`. CU-2 added
    // the ADR-040 §2.5-mandated float classification predicates `flt.is_nan`/`flt.is_finite`/
    // `flt.is_infinite` (unary `Float → Binary{1}`, `Empirical`). CU-6 added the bit-manipulation
    // counts `bit.popcount`/`bit.clz`/`bit.ctz` (unary `Binary{N} → Binary{N}`, `Exact`), bringing
    // Π to 66. CU-3 (ADR-040 §2.4) added the never-silent Binary↔Float conversions `bin.to_flt`
    // (checked-exact, `Exact`) and `flt.to_bin` (the width-witness shape, `Exact`), bringing Π to
    // 68. (This supersedes the DN-56/DN-76 "Π = 38" figure, which predates the M-887…M-899 +
    // CU-1/CU-2/CU-3/CU-6 landings — see DN-34 §8.15/§8.16; those docs are FLAGged for a count
    // refresh.)
    assert_eq!(t.entries().len(), 68);
}

// M-890 (`enb` Gap C): the dense elementwise group — the first non-`Exact` intrinsics in Π.
// The tags are carried from the kernel's `DenseSpace::op_guarantee` (VR-5: never upgraded):
// `neg` is `Exact` (negation never rounds on the symmetric dtype grids); `add`/`sub`/`scale`
// are `Proven` (Higham 2002 Thm 2.2, side-conditions checked per element by the kernel), and
// (M-891) the measurement pair `dense.dot`/`dense.similarity` is `Proven` (the binary64
// accumulation bound) with width-`Collapse` (two Dense{d, s} operands reduce to a Dense{1, F64}
// measurement — the tensor analogue of the cmp prims' reduce-to-Bool). The cross-crate
// consistency with `mycelium-dense` itself is guarded in `mycelium-interp` (this
// crate cannot depend on `mycelium-dense`); this test pins the table side of that contract.
#[test]
fn dense_group_carries_the_kernel_tags() {
    let t = PrimTable::builtins();
    for (name, arity, intrinsic, width) in [
        ("dense.add", 2, GuaranteeStrength::Proven, WidthRel::Uniform),
        ("dense.sub", 2, GuaranteeStrength::Proven, WidthRel::Uniform),
        ("dense.neg", 1, GuaranteeStrength::Exact, WidthRel::Uniform),
        (
            "dense.scale",
            2,
            GuaranteeStrength::Proven,
            WidthRel::Uniform,
        ),
        (
            "dense.dot",
            2,
            GuaranteeStrength::Proven,
            WidthRel::Collapse,
        ),
        (
            "dense.similarity",
            2,
            GuaranteeStrength::Proven,
            WidthRel::Collapse,
        ),
    ] {
        let d = t.get(name).expect("dense builtin registered");
        assert_eq!(d.intrinsic, intrinsic, "{name}: intrinsic tag drifted");
        assert_eq!(d.sig.arity(), arity, "{name}: arity drifted");
        assert_eq!(d.sig.width, width, "{name}: width relation drifted");
        // Paradigm-model escape hatch (FLAG in builtins()): Dense operands/results are `Any`
        // until a first-class Dense paradigm lands; the real typing is kernel + L1-checker.
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "{name}: operands are the documented `Any` escape hatch"
        );
        assert_eq!(d.sig.result, PrimParadigm::Any);
    }
}

// M-898 (`enb` Gap A): the scalar-float arithmetic group — the first `Empirical` intrinsics in Π.
// The tag is the ratified ADR-040 §2.6 posture (VR-5: never upgraded without a checked basis):
// the op's *definition* is the correctly-rounded IEEE-754 binary64 RNE result (`Exact` as a
// definition), the *implementation claim* that host f64 delivers that bit pattern is `Empirical`
// (pinned by the reference-case corpus in `mycelium-interp`), and the platform IEEE statement is
// `Declared`. No `Proven` anywhere (no checked side-condition theorem is claimed). This test pins
// the table side; the runtime-value tag/bound contract is guarded in `mycelium-interp`.
#[test]
fn flt_group_carries_the_adr040_empirical_intrinsic() {
    let t = PrimTable::builtins();
    for (name, arity) in [
        ("flt.add", 2),
        ("flt.sub", 2),
        ("flt.mul", 2),
        ("flt.div", 2),
        ("flt.neg", 1),
    ] {
        let d = t.get(name).expect("flt builtin registered");
        assert_eq!(
            d.intrinsic,
            GuaranteeStrength::Empirical,
            "{name}: intrinsic must stay the ratified ADR-040 §2.6 `Empirical` (VR-5)"
        );
        assert_eq!(d.sig.arity(), arity, "{name}: arity drifted");
        assert_eq!(d.sig.width, WidthRel::Uniform, "{name}: width rel drifted");
        // Paradigm-model escape hatch (FLAG in builtins()): no first-class `Float` paradigm yet;
        // the real all-operands-Float typing is the interpreter prim + the L1 checker branch.
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "{name}: operands are the documented `Any` escape hatch"
        );
        assert_eq!(d.sig.result, PrimParadigm::Any);
    }
}

// M-899 (`enb` Gap A): the scalar-float comparison group — explicit NaN semantics per the
// ratified ADR-040 §2.4. `flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` are the IEEE-754 §5.11
// partial-order *predicates* (any comparison involving NaN is the defined value false —
// `flt.eq(NaN, NaN) = false`), and `flt.total_le` is the **named, opt-in** total order
// (IEEE-754 §5.10 `totalOrder`). The intrinsic is `Empirical` per ADR-040 §2.6 (VR-5: never
// upgraded) — and, load-bearing here, the `totalOrder` total-order *property* stays `Empirical`
// **until the M-511 proof debt is discharged**: no `Proven` is claimed anywhere (no checked
// side-condition theorem exists). This test pins the table side; the runtime NaN-semantics
// corpus + property evidence lives in `mycelium-interp`.
#[test]
fn flt_cmp_group_carries_the_adr040_empirical_intrinsic() {
    let t = PrimTable::builtins();
    for name in [
        "flt.lt",
        "flt.le",
        "flt.gt",
        "flt.ge",
        "flt.eq",
        "flt.total_le",
    ] {
        let d = t.get(name).expect("flt comparison builtin registered");
        assert_eq!(
            d.intrinsic,
            GuaranteeStrength::Empirical,
            "{name}: intrinsic must stay the ratified ADR-040 §2.6 `Empirical` (VR-5; for \
             flt.total_le the total-order property is the M-511 proof debt — never `Proven` \
             without the checked theorem)"
        );
        assert_eq!(d.sig.arity(), 2, "{name}: arity drifted");
        assert_eq!(
            d.sig.width,
            WidthRel::Collapse,
            "{name}: width-collapsing (two Float scalars → Binary{{1}})"
        );
        // Operands stay the documented `Any` escape hatch (no first-class `Float` paradigm yet);
        // the result genuinely is a Binary{1} truth value, so `Binary` is precise.
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "{name}: operands are the documented `Any` escape hatch"
        );
        assert_eq!(d.sig.result, PrimParadigm::Binary);
    }
}

// M-892 (`enb` Gap C): the model-dispatched VSA bind group (RFC-0003 §3/§4; ADR-008). A Π
// declaration stores ONE intrinsic, but the per-op tag is per-*model* (MAP-I/BSC bind/unbind/
// permute `Exact`; FHRR bind/permute `Exact`, unbind `Empirical` — the RFC-0003 §4 normative
// weak-link assignment), so the table records the **meet over the dispatch set** — never the
// strongest member (VR-5: downgrade to stay accurate): `vsa.bind`/`vsa.permute` `Exact`,
// `vsa.unbind` `Empirical`. The runtime value still carries the dispatched model's own kernel
// tag; the table↔kernel meet consistency is guarded in `mycelium-interp` (this crate cannot
// depend on `mycelium-vsa` — ADR-008 keeps the dependency one-way). This test pins the table
// side. Widening the dispatch set (HRR/MAP-B/SBC, later waves) must recompute these meets.
#[test]
fn vsa_bind_group_carries_the_dispatch_set_meet_tags() {
    let t = PrimTable::builtins();
    for (name, intrinsic) in [
        ("vsa.bind", GuaranteeStrength::Exact),
        ("vsa.unbind", GuaranteeStrength::Empirical),
        ("vsa.permute", GuaranteeStrength::Exact),
    ] {
        let d = t.get(name).expect("vsa builtin registered");
        assert_eq!(
            d.intrinsic, intrinsic,
            "{name}: intrinsic must stay the meet over the MAP-I/FHRR/BSC dispatch set (VR-5)"
        );
        assert_eq!(d.sig.arity(), 2, "{name}: arity drifted");
        assert_eq!(d.sig.width, WidthRel::Uniform, "{name}: width rel drifted");
        // Paradigm-model escape hatch (FLAG in builtins()): no first-class `Vsa` paradigm yet;
        // the real equal-model+dim typing (and vsa.permute's Binary{W} shift operand) is the
        // interpreter prim + the L1 checker branch (`try_check_vsa_prim`).
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "{name}: operands are the documented `Any` escape hatch"
        );
        assert_eq!(d.sig.result, PrimParadigm::Any);
    }
}

// M-893 (`enb` Gap C): `vsa.bundle` — superposition via the **certified path** (RFC-0003 §4/§5;
// ADR-008). Its dispatch set is the certified singleton {MAP-I} (the only introduction-set model
// with a certified Value-level bundle — `bundle_values_certified`, the M-131 checked-instantiation
// pattern); FHRR/BSC bundles are `Empirical`-profile kernel ops, refused explicitly by the
// wrapper/checker rather than silently re-tagged (VR-5) — surfacing them is a distinct append-only
// extension. The intrinsic is therefore the meet over that singleton = MAP-I's `Bundle` tag =
// `Proven`; the runtime value carries the kernel-checked `CapacityBound` itself. The table↔kernel
// consistency is guarded in `mycelium-interp` (this crate cannot depend on `mycelium-vsa` —
// ADR-008 keeps the dependency one-way); this test pins the table side. Operands are the
// documented `Any`/`Uniform` escape hatch (really `Seq{Vsa{m, d}, N≥1}` × `Float` δ → `Vsa{m, d}`,
// enforced by the interpreter prim + the L1 checker branch).
#[test]
fn vsa_bundle_carries_the_certified_singleton_proven_tag() {
    let t = PrimTable::builtins();
    let d = t.get("vsa.bundle").expect("vsa.bundle registered");
    assert_eq!(
        d.intrinsic,
        GuaranteeStrength::Proven,
        "vsa.bundle: intrinsic must stay the meet over the certified singleton {{MAP-I}} (VR-5)"
    );
    assert_eq!(
        d.sig.arity(),
        2,
        "vsa.bundle: arity drifted (Seq + Float δ)"
    );
    assert_eq!(
        d.sig.width,
        WidthRel::Uniform,
        "vsa.bundle: width rel drifted"
    );
    assert!(
        d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
        "vsa.bundle: operands are the documented `Any` escape hatch"
    );
    assert_eq!(d.sig.result, PrimParadigm::Any);
}

// M-894 (`enb` Gap C): the cleanup/reconstruction pair + the capacity query (RFC-0003 §3/§5/§6;
// ADR-008; FR-S4). `vsa.cleanup` and `vsa.reconstruct` are exhaustive arg-max retrieval decodes
// guarded by the RFC-0010 §4.4 identifiability refusal, so their intrinsic is `Exact` — the meet
// over their dispatch sets (MAP-I/FHRR/BSC for cleanup, whose decision procedure is model-generic;
// {MAP-I, BSC} for reconstruct, whose unbind step is `Exact` self-inverse algebra in both — FHRR
// reconstruct is an explicit refusal, its `Empirical` unbind profile covers only single bind
// products). A non-`Exact` query/record passes its own (strength, bound) pair through to the
// runtime value via the RFC-0001 §4.7 meet — the intrinsic here is the op's own contribution,
// never the composed floor. `vsa.required_dim` is the M-131 checked instantiation of the cited
// capacity theorem (`proven_capacity_bound` — the side-condition holds by construction at the
// returned dim), so its intrinsic is `Proven`, the same stance as `vsa.bundle`. The table↔kernel
// consistency is guarded in `mycelium-interp` (ADR-008 keeps the dependency one-way); this test
// pins the table side.
#[test]
fn vsa_cleanup_reconstruct_and_capacity_query_carry_their_meet_tags() {
    let t = PrimTable::builtins();
    for (name, arity, intrinsic) in [
        ("vsa.cleanup", 2, GuaranteeStrength::Exact),
        ("vsa.reconstruct", 4, GuaranteeStrength::Exact),
        ("vsa.required_dim", 2, GuaranteeStrength::Proven),
    ] {
        let d = t.get(name).expect("vsa builtin registered");
        assert_eq!(
            d.intrinsic, intrinsic,
            "{name}: intrinsic must stay the meet over its dispatch set / the checked \
             instantiation stance (VR-5)"
        );
        assert_eq!(d.sig.arity(), arity, "{name}: arity drifted");
        assert_eq!(d.sig.width, WidthRel::Uniform, "{name}: width rel drifted");
        // Paradigm-model escape hatch (FLAG in builtins()): no first-class `Vsa` paradigm yet;
        // the real typing (equal model+dim hypervectors, the Seq codebook, the Float threshold/δ,
        // the Binary items/dim) is the interpreter prim + the L1 checker branch.
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Any),
            "{name}: operands are the documented `Any` escape hatch"
        );
        assert_eq!(d.sig.result, PrimParadigm::Any);
    }
}

// M-767 (`enb` Gap B): the signedness-split signed op set (RFC-0033 §4.1.2/§4.1.3; ADR-028).
// `bin.div_s`/`bin.rem_s`/`bin.shr_s` are width-uniform Binary×Binary→Binary like their unsigned
// counterparts; `cmp.lt_s` is width-collapsing like `cmp.eq`/`cmp.lt` but with its operands pinned
// `Binary` (not the D1 `Any` — balanced ternary's D1 order is already the signed order, so no
// ternary `lt_s` exists). All `Exact` (total/decidable over the in-range domain; div-by-zero /
// shift-range / the `min ÷ −1` overflow are runtime refusals, same posture as the unsigned pair).
#[test]
fn signed_split_ops_are_declared_with_pinned_signatures() {
    let t = PrimTable::builtins();
    for name in ["bin.div_s", "bin.rem_s", "bin.shr_s"] {
        let d = t.get(name).expect("signed-split builtin registered");
        assert_eq!(d.intrinsic, GuaranteeStrength::Exact, "{name}: intrinsic");
        assert_eq!(d.sig.arity(), 2, "{name}: arity");
        assert!(
            d.sig.operands.iter().all(|p| *p == PrimParadigm::Binary),
            "{name}: operands are Binary"
        );
        assert_eq!(d.sig.result, PrimParadigm::Binary, "{name}: result");
        assert_eq!(d.sig.width, WidthRel::Uniform, "{name}: width-uniform");
    }
    let d = t.get("cmp.lt_s").expect("cmp.lt_s registered");
    assert_eq!(d.intrinsic, GuaranteeStrength::Exact);
    assert_eq!(d.sig.arity(), 2);
    assert!(
        d.sig.operands.iter().all(|p| *p == PrimParadigm::Binary),
        "cmp.lt_s operands are pinned Binary (no ternary signed order exists)"
    );
    assert_eq!(d.sig.result, PrimParadigm::Binary);
    assert_eq!(
        d.sig.width,
        WidthRel::Collapse,
        "cmp.lt_s is width-collapsing (equal-width operands → Binary{{1}})"
    );
}

#[test]
fn build_is_deterministic() {
    // Two independent builds produce the same hashes (content-addressing is a pure function).
    assert_eq!(PrimTable::builtins(), PrimTable::builtins());
}

// Mutant-witness (prim.rs:71:9): PrimSig::arity() must return operands.len(), not 0 or 1.
// Tests against unary, binary, and zero-arity signatures to cover all three replacement constants.
#[test]
fn arity_reflects_operand_count() {
    // Zero-arity (no operands).
    let zero = PrimSig {
        operands: vec![],
        result: PrimParadigm::Any,
        width: WidthRel::Uniform,
    };
    assert_eq!(zero.arity(), 0);
    // Unary.
    let unary = PrimSig {
        operands: vec![PrimParadigm::Binary],
        result: PrimParadigm::Binary,
        width: WidthRel::Uniform,
    };
    assert_eq!(unary.arity(), 1);
    // Binary.
    let binary = PrimSig {
        operands: vec![PrimParadigm::Ternary, PrimParadigm::Ternary],
        result: PrimParadigm::Ternary,
        width: WidthRel::Uniform,
    };
    assert_eq!(binary.arity(), 2);
    // From builtins: core.id is unary, bit.xor is binary.
    let t = PrimTable::builtins();
    assert_eq!(t.get("core.id").unwrap().sig.arity(), 1);
    assert_eq!(t.get("bit.xor").unwrap().sig.arity(), 2);
}

// Mutant-witness (prim.rs:122:9): Display for PrimRef must emit a non-empty, `#`-prefixed
// string, not Ok(Default::default()) (which would emit nothing).
#[test]
fn prim_ref_display_is_hash_prefixed() {
    let t = PrimTable::builtins();
    let r = t.prim_ref("bit.xor").unwrap();
    let s = r.to_string();
    // Must start with `#` (the Unison-style prim reference spelling).
    assert!(
        s.starts_with('#'),
        "PrimRef display must start with '#': got {s:?}"
    );
    // Must be non-empty and carry the algo prefix from the hash.
    assert!(
        s.len() > 1,
        "PrimRef display must be non-trivial: got {s:?}"
    );
}

// Mutant-witness (prim.rs:191:9 and prim.rs:203:9): decl_hash and decl must return the
// actual registered entry for a known name, not always None.
#[test]
fn decl_hash_and_decl_return_entries_for_known_names() {
    let t = PrimTable::builtins();
    // decl_hash returns Some for a registered name.
    let h = t.decl_hash("bit.not");
    assert!(
        h.is_some(),
        "decl_hash must return Some for a registered prim"
    );
    // decl resolves the hash to the actual declaration.
    let d = t.decl(h.unwrap());
    assert!(d.is_some(), "decl must resolve a registered hash");
    assert_eq!(d.unwrap().intrinsic, GuaranteeStrength::Exact);
    // Unknown names return None.
    assert!(t.decl_hash("nonexistent").is_none());
}

// Mutant-witness (prim.rs:228:9 both true/false replacements): contains() must return true
// for registered names and false for unregistered names — both sides kill both replacements.
#[test]
fn contains_returns_true_iff_registered() {
    let t = PrimTable::builtins();
    assert!(
        t.contains("trit.mul"),
        "contains must be true for a registered prim"
    );
    assert!(t.contains("bit.and"));
    assert!(
        !t.contains("nonexistent"),
        "contains must be false for an unknown prim"
    );
    assert!(!t.contains(""));
}

// Mutant-witness (prim.rs:234:9 all three replacements: vec![], vec![""], vec!["xyzzy"]):
// names() must return exactly the registered kernel names, sorted — neither empty, nor
// containing blank strings, nor containing sentinel strings.
#[test]
fn names_returns_registered_sorted_names() {
    let t = PrimTable::builtins();
    let ns = t.names();
    // Exactly 54 builtins (the original 9 + RFC-0032 cmp.eq/cmp.lt/bit.add/bit.sub + D3
    // seq.len/seq.get + D4 bytes.len/get/slice/concat + DN-41 bit.width_cast + DN-58/M-817
    // fuse_join:binary + RFC-0033/M-887 bin.mul + RFC-0033/M-888 bin.div/bin.rem + RFC-0033/M-889
    // bin.shl/bin.shr + RFC-0033/M-766 bin.add/bin.sub/bin.neg + RFC-0001 §4.1/M-890
    // dense.add/dense.sub/dense.neg/dense.scale + M-891 dense.dot/dense.similarity +
    // ADR-040 §2.5/M-898 flt.add/flt.sub/flt.mul/flt.div/flt.neg + ADR-040 §2.4/M-899
    // flt.lt/flt.le/flt.gt/flt.ge/flt.eq/flt.total_le + RFC-0033/M-767
    // bin.div_s/bin.rem_s/bin.shr_s/cmp.lt_s, the signedness-split signed set + RFC-0003
    // §3/§4/M-892 vsa.bind/vsa.unbind/vsa.permute, the model-dispatched VSA bind group +
    // RFC-0003 §4/§5/M-893 vsa.bundle, the certified superposition path + RFC-0003
    // §3/§5/§6/M-894 vsa.cleanup/vsa.reconstruct/vsa.required_dim, the cleanup/reconstruction
    // pair and the capacity-bound query + M-912 bytes.eq/hash.blake3, the folded-in byte
    // equality gap and the kernel's BLAKE3 content-addressing hash surfaced as a prim + CU-1
    // bit.mul, the never-silent unsigned multiply — RFC-0033 §4.1.2, the math.myc FLAG-math-1
    // missing op + CU-2 flt.is_nan/flt.is_finite/flt.is_infinite, the ADR-040 §2.5-mandated float
    // classification predicates + CU-6 bit.popcount/bit.clz/bit.ctz — bringing Π to 66 + CU-3
    // bin.to_flt/flt.to_bin, the never-silent Binary↔Float conversions (ADR-040 §2.4) — bringing
    // Π to 68).
    assert_eq!(
        ns.len(),
        68,
        "names() count must match the builtin count: {ns:?}"
    );
    // Sorted (BTreeMap iteration is sorted).
    let mut sorted = ns.clone();
    sorted.sort();
    assert_eq!(ns, sorted, "names() must be in sorted order");
    // Must contain specific known names, not blank/sentinel strings.
    assert!(ns.contains(&"bit.xor"), "must contain 'bit.xor'");
    assert!(ns.contains(&"core.id"), "must contain 'core.id'");
    assert!(!ns.contains(&""), "must not contain empty string");
    assert!(!ns.contains(&"xyzzy"), "must not contain sentinel 'xyzzy'");
}
