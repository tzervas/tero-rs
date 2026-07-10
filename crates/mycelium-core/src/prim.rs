//! The **prim table `Π`** as content-addressed declarations — RFC-0007 §4.4 (T-Op); RFC-0007 §8
//! R7-Q4; DN-10 §3; ADR-003 (Unison identity); RFC-0001 §4.7 (the intrinsic guarantee `g_f`). M-390.
//!
//! RFC-0007 §4.4 gives every primitive `p` a signature `Π(p) = (τ₁…τₙ) → τ` and (RFC-0001 §4.7) an
//! *intrinsic guarantee* `g_f` it contributes to the result's guarantee meet. Historically `Π` was a
//! fixed builtin table hard-coded in the elaborator/typechecker and an `Exact` constant in the
//! interpreter. Here it becomes **declarations with their own content addresses** — exactly the model
//! the data registry `Σ` ([`crate::data::DataRegistry`]) already uses for constructors (RFC-0001
//! §4.3 r3): each prim is keyed by the content hash of its *signature + intrinsic guarantee*, with
//! its (kernel) name kept separately as metadata (ADR-003 — names are not identity). A prim is then
//! an inspectable, EXPLAIN-able registry entry (G2/SC-3), not a black box.
//!
//! # Scope (honesty)
//! Nearly every builtin is `intrinsic = Exact` (the exact, elementwise/arithmetic fragment). The
//! table stores that intrinsic *as data* so a non-`Exact` prim is a registry entry carrying its own
//! honest tag — and the **dense elementwise group** (`dense.add`/`dense.sub`/`dense.scale`, M-890,
//! `enb` Gap C) is the first to use that capacity: their intrinsic is **`Proven`**, carried
//! verbatim from the kernel's per-op tag (`mycelium-dense`'s `DenseSpace::op_guarantee` — the
//! round-to-nearest relative-error theorem with per-element *checked* side-conditions; `dense.neg`
//! stays `Exact`, negation never rounds). The **dense measurement pair**
//! (`dense.dot`/`dense.similarity`, M-891) is likewise `Proven`, its bound the binary64
//! *accumulation* theorem (absolute/`Linf`, dimension-dependent) rather than the dtype's
//! per-element `op_rel_eps` — see the entry comment in [`PrimTable::builtins`]. *How* a non-`Exact` prim's bound-basis is stored **with
//! the declaration** (a cited theorem vs an empirical fit, with its [`crate::BoundBasis`]) is the
//! **RP-7** spike (DN-10 §3.6), deliberately *not* settled here — the declaration stores the
//! *strength* only, and the checked basis (theorem citation + per-element ε) rides the runtime
//! result `Value`'s `Meta`, attached by the kernel itself (`mycelium-dense::DenseSpace`), never
//! fabricated at the table level (VR-5). This crate cannot depend on `mycelium-dense` (dependency
//! direction), so the table↔kernel tag consistency is guarded by a test in `mycelium-interp`
//! (which sees both).
//!
//! The migration preserves `Π`-lookup semantics exactly: for every prim `p`,
//! `Π_new(hash(p)) = Π_old(name(p))` (DN-10 §3.4) — guarded by the `Π_new == Π_old` equivalence
//! tests in `mycelium-l1` (the surface table) and `mycelium-interp` (the intrinsic).

use std::collections::BTreeMap;

use crate::content::Canon;
use crate::guarantee::GuaranteeStrength;
use crate::id::ContentHash;

/// The representation paradigm of a prim operand or result (the `τ`'s paradigm in `Π(p)`). `Any` is
/// the paradigm-polymorphic identity (`core.id : a → a`); the concrete paradigms pin a prim to
/// `Binary{·}` or `Ternary{·}` (RFC-0007 §4.4). Width is governed separately by [`WidthRel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimParadigm {
    /// Paradigm-polymorphic (the identity prim): any single paradigm, passed through.
    Any,
    /// A `Binary{n}` operand/result.
    Binary,
    /// A `Ternary{m}` operand/result.
    Ternary,
}

/// How a prim's operand and result *widths* relate. Most of the builtin set is width-preserving —
/// every operand and the result share one width (`bit.xor : Binary{n} × Binary{n} → Binary{n}`,
/// `trit.add : Ternary{m} × Ternary{m} → Ternary{m}`, the unary cases trivially: [`WidthRel::Uniform`]).
/// The reduce-to-`Bool` comparison prims (`cmp.eq`/`cmp.lt`, RFC-0032 D1) are the exception — they
/// **collapse** to a fixed `Binary{1}` independent of the operand width ([`WidthRel::Collapse`]). New
/// rules (e.g. a width-changing pack) are added as variants, never silently assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidthRel {
    /// All operands and the result share one width.
    Uniform,
    /// The result width is **fixed and independent** of the operands' shared width — the
    /// width-collapsing rule of the reduce-to-`Bool` comparison prims (`cmp.eq`/`cmp.lt`, RFC-0032
    /// D1): two equal-width operands reduce to a one-bit `Binary{1}` truth value. (Operand widths
    /// must still agree; only the result is decoupled.)
    Collapse,
}

/// A prim's signature `Π(p) = (τ₁…τₙ) → τ` (RFC-0007 §4.4): the per-operand paradigms (arity is their
/// count), the result paradigm, and the width relation. Identity-bearing; names are excluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimSig {
    /// Operand paradigms, in order (the length is the prim's arity).
    pub operands: Vec<PrimParadigm>,
    /// The result paradigm.
    pub result: PrimParadigm,
    /// How operand/result widths relate.
    pub width: WidthRel,
}

impl PrimSig {
    /// The prim's arity (operand count).
    #[must_use]
    pub fn arity(&self) -> usize {
        self.operands.len()
    }
}

/// A resolved, content-addressed prim declaration: its signature and the *intrinsic guarantee* `g_f`
/// it contributes to a result's guarantee meet (RFC-0001 §4.7). The (kernel) name is stored
/// separately in the [`PrimTable`] (it is not identity — ADR-003).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimDecl {
    /// The signature `(τ₁…τₙ) → τ`.
    pub sig: PrimSig,
    /// The intrinsic guarantee `g_f` (RFC-0001 §4.7). `Exact` for every v0 builtin.
    pub intrinsic: GuaranteeStrength,
}

impl PrimDecl {
    /// The content hash of this declaration's identity-bearing content (signature + intrinsic
    /// guarantee), with the name excluded (ADR-003). Two prims are the *same* prim iff their
    /// signature and intrinsic agree — domain-separated from node/data hashes so a prim can never
    /// collide with a structural node hash.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        let mut c = Canon::new();
        c.prim_decl(&self.sig, self.intrinsic);
        c.finish()
    }
}

/// A prim reference `#p` (the prim analogue of [`CtorRef`](crate::CtorRef) `#T#i`): the content hash
/// of a [`PrimDecl`]. A term referring to a prim by *identity* refers to it by this hash, not its
/// name (ADR-003).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimRef(ContentHash);

impl PrimRef {
    /// Build a prim reference from a declaration hash.
    #[must_use]
    pub fn new(decl: ContentHash) -> Self {
        PrimRef(decl)
    }

    /// The referenced declaration's content hash.
    #[must_use]
    pub fn decl(&self) -> &ContentHash {
        &self.0
    }
}

impl core::fmt::Display for PrimRef {
    /// The Unison-style spelling `#<declhash>` (a prim has no constructor index, unlike `#T#i`).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0.as_str())
    }
}

/// The content-addressed **prim table `Π`** (RFC-0007 §4.4; R7-Q4): resolved declarations keyed by
/// their content hash, plus the build-time `name → hash` resolution used to form [`PrimRef`]s — the
/// same two-map shape as [`DataRegistry`](crate::DataRegistry), so a prim's identity (`#p`) is the
/// same on every path (the NFR-7 differential is over *one* prim set, never two).
///
/// Prims have no inter-references (unlike data, which can be mutually recursive), so building is a
/// flat hash-and-insert — no SCC/cycle handling is needed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrimTable {
    /// Resolved declarations, keyed by content hash.
    decls: BTreeMap<ContentHash, PrimDecl>,
    /// Build-time (kernel) name → content hash (names are metadata — ADR-003).
    by_name: BTreeMap<String, ContentHash>,
}

impl PrimTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        PrimTable::default()
    }

    /// Register (or replace) a prim declaration under build-time kernel name `name`, returning its
    /// [`PrimRef`]. Re-registering a name re-points it; identity is the decl hash, not the name.
    pub fn insert(&mut self, name: impl Into<String>, decl: PrimDecl) -> PrimRef {
        let hash = decl.content_hash();
        self.by_name.insert(name.into(), hash.clone());
        self.decls.insert(hash.clone(), decl);
        PrimRef::new(hash)
    }

    /// The default table: the closed v0 kernel-prim set — the identity, the elementwise binary logic
    /// (`bit.*`), the fixed-width balanced-ternary arithmetic (`trit.*`, M-111), the reduce-to-`Bool`
    /// comparison prims (`cmp.eq`/`cmp.lt`, RFC-0032 D1), the never-silent binary arithmetic
    /// (`bit.add`/`bit.sub`, RFC-0032 D2), the never-silent two's-complement multiply
    /// (`bin.mul`, RFC-0033 §4.1.2/§4.1.3, M-887 — the first Gap-B `enb` prim), the never-silent
    /// **unsigned** division/remainder (`bin.div`/`bin.rem`, RFC-0033 §4.1.2/§4.1.3, M-888), the
    /// never-silent **logical** left/right shift (`bin.shl`/`bin.shr`, RFC-0033 §4.1.2/§4.1.3,
    /// M-889 — the signed/arithmetic variants ride M-767 under distinct names), and the never-silent
    /// two's-complement `add`/`sub`/`neg` (`bin.add`/`bin.sub`/`bin.neg`, RFC-0033 §4.1.2/§4.1.3,
    /// M-766 — completes the shared two's-complement op set `add`/`sub`/`mul`/`neg`; distinct from
    /// the pre-existing unsigned `bit.add`/`bit.sub`, which under-refuse relative to the signed
    /// domain), and the **signedness-split signed op set** (`bin.div_s`/`bin.rem_s`/`bin.shr_s` +
    /// `cmp.lt_s`, RFC-0033 §4.1.2/§4.1.3, M-767 — signed truncated division/remainder, the
    /// arithmetic right shift, and the two's-complement ordering, each a distinct named op from
    /// its unsigned counterpart per ADR-028), and the **dense elementwise group**
    /// (`dense.add`/`dense.sub`/`dense.neg`/`dense.scale`, RFC-0001 §4.1/RFC-0002 §5, M-890 —
    /// `enb` Gap C; the first *tensor-valued* prims, and the first non-`Exact` intrinsics — see
    /// the crate-level Scope note), plus the **dense measurement pair**
    /// (`dense.dot`/`dense.similarity`, M-891 — two `Dense{d, s}` operands reduce to a
    /// `Dense{1, F64}` measurement carrying the kernel's proven binary64 accumulation bound),
    /// and the **scalar-float arithmetic group**
    /// (`flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg`, ADR-040 §2.5, M-898 — `enb` Gap A;
    /// IEEE-754 binary64 under RNE over `Repr::Float`, in-band specials per the ratified FLAG-2),
    /// and the **scalar-float comparison group**
    /// (`flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` — the IEEE-754 §5.11 partial-order
    /// predicates, NaN explicitly unordered — plus the named opt-in total order `flt.total_le`
    /// (IEEE-754 §5.10 `totalOrder`), ADR-040 §2.4, M-899 — `enb` Gap A; the total-order
    /// *property* stays `Empirical` until the M-511 proof debt is discharged),
    /// and the **VSA bind group** (`vsa.bind`/`vsa.unbind`/`vsa.permute`, RFC-0003 §3/§4/ADR-008,
    /// M-892 — `enb` Gap C; model-dispatched MAP-I/FHRR/BSC, tags per model carried from the
    /// `mycelium-vsa` kernel — see the entry comment for the meet-tag rule), plus the **certified
    /// VSA superposition** (`vsa.bundle`, RFC-0003 §4/§5/ADR-008, M-893 — `enb` Gap C; the
    /// certified path via MAP-I's `bundle_values_certified`, dispatch set the certified singleton
    /// {MAP-I}; the runtime value carries the kernel-checked `Proven` `CapacityBound` — see the
    /// entry comment), and the **VSA cleanup/reconstruction pair + capacity query**
    /// (`vsa.cleanup`/`vsa.reconstruct`/`vsa.required_dim`, RFC-0003 §3/§5/§6/ADR-008, M-894 —
    /// `enb` Gap C; the FR-S4 cleanup-memory retrieval returning the `[index, confidence,
    /// margin]` decision triple, the §6 compositional role-reconstruction with an explicit
    /// threshold, and the M-131 `requiredDim`/`proven_capacity_bound` query — see the entry
    /// comment), plus **`bytes.eq`** (M-912, `enb` — the folded-in equality gap the diag/error/
    /// recover ports flagged: byte-wise equality over two `Bytes` operands) and **`hash.blake3`**
    /// (M-912, `enb` — the kernel's own BLAKE3 content-addressing hash, M-103, surfaced as a
    /// `Bytes -> Bytes` prim; `Exact`, justified by the kernel's own deterministic use).
    /// Every entry is `intrinsic = Exact` **except** `dense.add`/`dense.sub`/`dense.scale`/
    /// `dense.dot`/`dense.similarity` (`Proven`, carried from the kernel's per-op tag), the
    /// `flt.*` group (`Empirical` — the ratified ADR-040 §2.6 host-conformance posture; see the
    /// entry comment), `vsa.unbind` (`Empirical` — the meet over the model set: FHRR's
    /// normative weak-link unbind; see the entry comment), `vsa.bundle` (`Proven` — the meet
    /// over its certified singleton dispatch set {MAP-I}; see the entry comment), and
    /// `vsa.required_dim` (`Proven` — the M-131 checked instantiation of the cited capacity
    /// theorem; see the entry comment); all are
    /// width-`Uniform` **except**
    /// `cmp.eq`/`cmp.lt`/`cmp.lt_s`,
    /// `dense.dot`/`dense.similarity`, and the `flt.*` comparison group, which are
    /// width-`Collapse` (operand width → `Binary{1}` / operand dim → a dim-1 measurement /
    /// two `Float` scalars → a `Binary{1}` truth value). This is the single source of truth the
    /// `mycelium-interp` intrinsic and the `mycelium-l1` surface table are checked against.
    #[must_use]
    pub fn builtins() -> Self {
        use PrimParadigm::{Any, Binary, Ternary};
        let mut t = PrimTable::new();
        let exact = |operands: Vec<PrimParadigm>, result: PrimParadigm| PrimDecl {
            sig: PrimSig {
                operands,
                result,
                width: WidthRel::Uniform,
            },
            intrinsic: GuaranteeStrength::Exact,
        };
        // RFC-0032 D1 (M-747): the reduce-to-`Bool` comparison prims are width-*collapsing* — two
        // equal-width operands of either paradigm reduce to a `Binary{1}` truth value. The operands
        // are typed `Any` because each may be Binary OR Ternary; this does NOT permit a *cross*-
        // paradigm comparison — the same-paradigm + equal-width constraint is enforced (never-silent,
        // G2) by the interpreter prim (`prims.rs::cmp_repr_operands`) and the L1 checker branch
        // (`checkty.rs`), since the per-operand `Any` paradigm model cannot express "both agree".
        let cmp = || PrimDecl {
            sig: PrimSig {
                operands: vec![Any, Any],
                result: Binary,
                width: WidthRel::Collapse,
            },
            intrinsic: GuaranteeStrength::Exact,
        };
        // Identity (paradigm-polymorphic passthrough).
        t.insert("core.id", exact(vec![Any], Any));
        // Elementwise binary logic.
        t.insert("bit.not", exact(vec![Binary], Binary));
        t.insert("bit.and", exact(vec![Binary, Binary], Binary));
        t.insert("bit.or", exact(vec![Binary, Binary], Binary));
        t.insert("bit.xor", exact(vec![Binary, Binary], Binary));
        // Fixed-width balanced-ternary arithmetic (M-111).
        t.insert("trit.neg", exact(vec![Ternary], Ternary));
        t.insert("trit.add", exact(vec![Ternary, Ternary], Ternary));
        t.insert("trit.sub", exact(vec![Ternary, Ternary], Ternary));
        t.insert("trit.mul", exact(vec![Ternary, Ternary], Ternary));
        // RFC-0032 D1 (M-747): reduce-to-`Bool` comparison/equality (width-collapsing → Binary{1}).
        t.insert("cmp.eq", cmp());
        t.insert("cmp.lt", cmp());
        // RFC-0032 D2 (M-748): never-silent fixed-width binary arithmetic (width-uniform).
        t.insert("bit.add", exact(vec![Binary, Binary], Binary));
        t.insert("bit.sub", exact(vec![Binary, Binary], Binary));
        // RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement `Binary`
        // multiply — the first landed op of the *shared* (signedness-agnostic bit-pattern)
        // two's-complement arithmetic set ADR-028 names (`add`/`sub`/`mul`/`neg`). `intrinsic =
        // Exact` (total/decidable over the in-range domain; an out-of-range product is a runtime,
        // not intrinsic, refusal — same posture as `bit.add`/`bit.sub`).
        t.insert("bin.mul", exact(vec![Binary, Binary], Binary));
        // RFC-0033 §4.1.2 (CU-1): never-silent **unsigned** `Binary` multiply — the unsigned member
        // of the `bit.*` family (overflow-distinct from signed `bin.mul` per §4.1.2), the multiply
        // `lib/std/math.myc` FLAG-math-1 named as missing. `intrinsic = Exact` (total/decidable over
        // the in-`U_N`-range domain; an out-of-range product is a runtime, not intrinsic, refusal).
        t.insert("bit.mul", exact(vec![Binary, Binary], Binary));
        // CU-6: width-preserving bit-manipulation counts (population count, count-leading-zeros,
        // count-trailing-zeros) — unary `Binary{N} → Binary{N}`, `Exact` (total/decidable; a count
        // always fits `N` bits, no runtime refusal). Kernel prims per KC-3 + performance (single host
        // instruction, not efficiently `.myc`-derivable); rotate/reverse_bits ride `std.math`.
        t.insert("bit.popcount", exact(vec![Binary], Binary));
        t.insert("bit.clz", exact(vec![Binary], Binary));
        t.insert("bit.ctz", exact(vec![Binary], Binary));
        // RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent **unsigned** `Binary`
        // division/remainder. Distinct-named from a future signed variant (M-767) per §4.1.2's
        // signedness-split requirement for division. `intrinsic = Exact` (total/decidable over the
        // nonzero-divisor domain; div-by-zero is a runtime, not intrinsic, refusal).
        t.insert("bin.div", exact(vec![Binary, Binary], Binary));
        t.insert("bin.rem", exact(vec![Binary, Binary], Binary));
        // RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent **logical** (unsigned) `Binary`
        // left/right shift — the third Gap-B prim of the signedness-split `shift` op set (§4.1.2).
        // Both operands are `Binary{N}` (the shift amount is itself read as an unsigned `N`-bit
        // bitvector); a shift amount `>= N` is a runtime, not intrinsic, refusal (never UB/wrap), so
        // `intrinsic = Exact` — same posture as `bin.div`/`bin.rem`'s div-by-zero. The **arithmetic**
        // (sign-extending) right shift is the distinct signed op M-767 lands under its own name.
        t.insert("bin.shl", exact(vec![Binary, Binary], Binary));
        t.insert("bin.shr", exact(vec![Binary, Binary], Binary));
        // RFC-0033 §4.1.2/§4.1.3 (M-766, `enb` Gap B): never-silent two's-complement `add`/`sub`/
        // `neg` — completes the *shared* two's-complement arithmetic set `bin.mul` (M-887) started.
        // Distinct from the pre-existing `bit.add`/`bit.sub` (RFC-0032 D2, unsigned-committed
        // overflow criterion — verified insufficient for the signed domain: e.g. `Binary{4}`'s
        // `5 + 3 = 8` is unsigned-in-range `[0,15]` but signed-out-of-range `B_4 = [-8,7]`).
        // `intrinsic = Exact` (total/decidable over the in-range domain; an out-of-range sum/
        // difference/negation is a runtime, not intrinsic, refusal — same posture as `bin.mul`).
        t.insert("bin.add", exact(vec![Binary, Binary], Binary));
        t.insert("bin.sub", exact(vec![Binary, Binary], Binary));
        t.insert("bin.neg", exact(vec![Binary], Binary));
        // RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the **signedness-split** op set — signed
        // (two's-complement) division/remainder and the arithmetic (sign-extending) right shift,
        // the distinct-named signed counterparts to `bin.div`/`bin.rem`/`bin.shr` (ADR-028:
        // signedness lives in the *op*, not the `Repr`; the SMT-LIB `bvsdiv`/`bvudiv`,
        // `bvashr`/`bvlshr` split). Division is truncated toward zero, remainder sign follows the
        // dividend (SMT-LIB `bvsdiv`/`bvsrem` — see `mycelium_core::binary`'s rounding-convention
        // note). `intrinsic = Exact` (total/decidable over the in-range domain; div-by-zero, an
        // out-of-range shift amount, and the single signed-division overflow `min ÷ −1` are
        // runtime, not intrinsic, refusals — same posture as the unsigned pair).
        t.insert("bin.div_s", exact(vec![Binary, Binary], Binary));
        t.insert("bin.rem_s", exact(vec![Binary, Binary], Binary));
        t.insert("bin.shr_s", exact(vec![Binary, Binary], Binary));
        // RFC-0033 §4.1.2 (M-767): the **signed** (two's-complement) ordering — `cmp.lt` reads
        // `Binary` operands as unsigned magnitudes (the D1 total order), so the signed order MUST
        // be a distinct named op (ADR-028's `bvslt`/`bvult` split). Width-collapsing like
        // `cmp.eq`/`cmp.lt` (two equal-width operands → a `Binary{1}` truth value) — but its
        // operands are pinned `Binary` (not the D1 pair's `Any`): balanced ternary is inherently
        // signed, so its D1 `cmp.lt` order IS the signed order and no ternary `lt_s` exists.
        t.insert(
            "cmp.lt_s",
            PrimDecl {
                sig: PrimSig {
                    operands: vec![Binary, Binary],
                    result: Binary,
                    width: WidthRel::Collapse,
                },
                intrinsic: GuaranteeStrength::Exact,
            },
        );
        // DN-41 (M-798): never-silent `Binary` width-cast (zero-extend widen / checked narrow).
        // `intrinsic = Exact` (the widen/identity/in-range-narrow result equals the unsigned value
        // exactly; a lossy narrow is a never-silent *runtime* refusal, not a non-Exact intrinsic).
        // **Width-model note (FLAG):** the Π `WidthRel` model is `Uniform`/`Collapse` only — it has
        // **no first-class width-*change* relation**, so this width-cast prim cannot express "result
        // width = the *second* (witness) operand's width" in the coarse table. It is recorded
        // `Uniform` here as the nearest tag; the real never-silent typing — both operands `Binary`,
        // result width = witness width `M`, the narrowing-fit refusal — is enforced by the interpreter
        // prim (`prims.rs::prim_width_cast`) and the L1 checker (`checkty.rs`), exactly as the seq/
        // bytes prims' real typing lives in their interpreter prims (same paradigm-model escape hatch).
        // A first-class width-change `WidthRel` is a deliberate, RFC-unpinned extension left for later.
        t.insert("bit.width_cast", exact(vec![Binary, Binary], Binary));
        // RFC-0032 D3 (M-749): never-silent indexed-sequence access. Both are `intrinsic = Exact`
        // (total/decidable over the in-range domain). **Paradigm-model note (FLAG):** the Π paradigm
        // model is `Binary`/`Ternary`/`Any` only — it has no first-class `Seq` paradigm, and a
        // sequence-element result type cannot be expressed in it. So the seq operand and the
        // `seq.get` element result are typed `Any` here (the table's existing paradigm-polymorphic
        // escape hatch, as for `core.id`); the real never-silent typing — "operand must be a `Seq`",
        // out-of-bounds refusal, the element repr of the result — is enforced by the interpreter prim
        // (`prims.rs::{as_seq,as_index,prim_seq_get}`), not encoded in this coarse signature. A
        // first-class `Seq` paradigm in `PrimParadigm` is a deliberate, RFC-unpinned extension left
        // for the surface-typing work (it ripples into the checker + content-addressing).
        t.insert("seq.len", exact(vec![Any], Binary));
        t.insert("seq.get", exact(vec![Any, Binary], Any));
        // RFC-0032 D4 (M-750): never-silent byte-string access. All `intrinsic = Exact`. Same
        // paradigm-model FLAG as the seq prims: the Π model has no first-class `Bytes` paradigm, so
        // the bytes operand/result are typed `Any` (the real "operand must be `Bytes`" + out-of-range
        // refusals are enforced by the interpreter prims `prims.rs::{as_bytes_payload,prim_bytes_*}`).
        // `bytes.len`/`bytes.get` produce a `Binary` (length / a `Binary{8}` byte); `bytes.slice`/
        // `bytes.concat` produce `Bytes` (typed `Any` here).
        t.insert("bytes.len", exact(vec![Any], Binary));
        t.insert("bytes.get", exact(vec![Any, Binary], Binary));
        t.insert("bytes.slice", exact(vec![Any, Binary, Binary], Any));
        t.insert("bytes.concat", exact(vec![Any, Any], Any));
        // M-912 (`enb`, folded-in gap): `bytes.eq` — byte-wise equality over two `Bytes` operands,
        // flagged missing by the diag/error/recover ports (`bytes.*` had len/get/slice/concat but no
        // equality). Same `Any`/`Binary` escape hatch as the rest of the group (no first-class
        // `Bytes` paradigm); the real "both operands `Bytes`" typing is the interpreter prim
        // (`prims.rs::prim_bytes_eq`) and the L1 checker branch. `intrinsic = Exact` — a total,
        // decidable `[u8]` comparison, no approximation involved.
        t.insert("bytes.eq", exact(vec![Any, Any], Binary));
        // M-912 (`enb`): `hash.blake3` — the kernel's own content-addressing hash (BLAKE3, M-103;
        // `mycelium-core::content::Canon`/`id::ContentHash` already use it) surfaced as a prim:
        // `Bytes -> Bytes`, the 32-byte digest of the input byte string. Same `Any` escape hatch (no
        // first-class `Bytes` paradigm); the real "operand must be `Bytes`" typing is the interpreter
        // prim (`prims.rs::prim_hash_blake3`) and the L1 checker branch. `intrinsic = Exact` —
        // justified by the kernel's own use of BLAKE3 for content addressing (deterministic; the
        // wrapper calls the same algorithm the same way, adding no additional uncertainty).
        t.insert("hash.blake3", exact(vec![Any], Any));
        // DN-58 §A (M-817): the `Binary` `Fuse` semilattice meet (bitwise-AND). `intrinsic = Exact`
        // (a total greatest-lower-bound). The user-`Data` fuse registers no prim — it elaborates to the
        // resolved `Fuse::join` call (DN-58 §A.5) — and the non-`Binary` reprs have no committed meet
        // (DN-58 §A.6 F-A3), so this is the only `fuse_join:*` kernel prim.
        t.insert("fuse_join:binary", exact(vec![Binary, Binary], Binary));
        // RFC-0001 §4.1 / RFC-0002 §5 (M-890, `enb` Gap C): the **dense elementwise group** —
        // the first *tensor-valued* prims (operands/results are `Repr::Dense{dim, dtype}` values).
        // Kernel: `mycelium-dense`'s `add_values`/`sub_values`/`neg_value`/`scale_value`.
        //
        // **Intrinsic tags — carried from the kernel, never upgraded (VR-5).** These mirror
        // `DenseSpace::op_guarantee` verbatim: `neg` is `Exact` (the dtype grids are symmetric —
        // negation never rounds); `add`/`sub`/`scale` are **`Proven`** — the round-to-nearest
        // relative-error theorem (Higham 2002, Thm 2.2) with side-conditions *checked per element*
        // by the kernel (exact on-grid inputs; finite, zero-or-normal, non-overflowing results —
        // a violated side-condition is an explicit runtime refusal, never a bound the theorem does
        // not cover). Per RP-7 (still open — see the crate Scope note) the declaration stores the
        // *strength* only; the checked basis (citation + per-element ε) rides the runtime result
        // `Value`, attached by the kernel. Consistency with `DenseSpace::op_guarantee` is guarded
        // by a `mycelium-interp` test (this crate cannot see `mycelium-dense`).
        //
        // **Paradigm/width-model note (FLAG — same escape hatch as the seq/bytes prims above):**
        // `PrimParadigm` has no first-class `Dense` paradigm and `WidthRel` no dim relation, so
        // the operands/results are typed `Any`/`Uniform` here as the nearest tags. The real
        // never-silent typing — `Dense{d, s}` operands with *equal* dim + dtype (shape mismatch is
        // an explicit refusal, never a broadcast), and `dense.scale`'s scalar operand as a
        // `Dense{1, s}` (the only float-bearing value form pre-Gap-A; see `prims.rs` in
        // `mycelium-interp`) — is enforced by the kernel + interpreter prim and the L1 checker. A
        // first-class `Dense` paradigm in `PrimParadigm` is a deliberate, RFC-unpinned extension
        // left for the surface-typing work (it ripples into content-addressing), exactly as for
        // `Seq`/`Bytes`.
        let dense_proven = |operands: Vec<PrimParadigm>| PrimDecl {
            sig: PrimSig {
                operands,
                result: Any,
                width: WidthRel::Uniform,
            },
            intrinsic: GuaranteeStrength::Proven,
        };
        t.insert("dense.add", dense_proven(vec![Any, Any]));
        t.insert("dense.sub", dense_proven(vec![Any, Any]));
        t.insert("dense.neg", exact(vec![Any], Any));
        t.insert("dense.scale", dense_proven(vec![Any, Any]));
        // RFC-0001 §4.1 / RFC-0002 §5 (M-891, `enb` Gap C): the **dense measurement pair** —
        // `dense.dot`/`dense.similarity` reduce two `Dense{d, s}` operands to a single
        // `Dense{1, F64}` measurement, so their width relation is `Collapse` (the tensor
        // analogue of `cmp.eq`/`cmp.lt`'s reduce-to-`Bool`; the result dim is fixed at 1,
        // independent of the operands' shared dim).
        //
        // **Intrinsic — `Proven`, carried from the kernel (`DenseSpace::op_guarantee`), and its
        // bound is the binary64 *accumulation* bound, NOT `op_rel_eps`:** over exact on-grid
        // F32/BF16 operands every product is exact in the f64 accumulator, so the dtype's
        // per-element rounding ε never enters, and a per-element *relative* claim on a dot
        // product would be false under cancellation. The honest disclosed ε is absolute (`Linf`)
        // and dimension-dependent (`DenseSpace::dot_abs_eps`/`similarity_abs_eps`), riding the
        // runtime result `Value` with its `ProvenThm` citation (RP-7 posture unchanged: the
        // declaration stores the strength only). Consistency with the kernel is guarded in
        // `mycelium-interp` (as for the M-890 group).
        let dense_measure = || PrimDecl {
            sig: PrimSig {
                operands: vec![Any, Any],
                result: Any,
                width: WidthRel::Collapse,
            },
            intrinsic: GuaranteeStrength::Proven,
        };
        t.insert("dense.dot", dense_measure());
        t.insert("dense.similarity", dense_measure());
        // ADR-040 §2.5 (M-898, `enb` Gap A): the **scalar-float arithmetic group** —
        // `flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg` over `Repr::Float{F64}` (IEEE-754
        // binary64, round-to-nearest-even only; rounding is a property of the *operation*, never
        // hidden state — ADR-040 §2.2, the ADR-028 parallel). Arithmetic specials (±inf, NaN) are
        // **in-band, inspectable, propagating values** (ADR-040 §2.4, ratified FLAG-2): overflow
        // → ±inf, div-by-zero → ±inf, 0/0 → NaN — never a trap and never a silent wrap onto an
        // ordinary in-range value; the distinguished in-band sentinel IS the never-silent signal
        // (dedicated classification prims `is_nan`/`is_finite` are still OPEN — M-899 shipped
        // comparison/total-order only; until they land, NaN is detectable as `¬flt.eq(x, x)` and
        // finiteness as `flt.lt(-inf, x) ∧ flt.lt(x, +inf)` — FLAGged, never silently dropped).
        // Every NaN result carries
        // the canonical bits (`Value::new` construction invariant, ADR-040 §2.3).
        //
        // **Intrinsic — `Empirical`, per the ratified ADR-040 §2.6 (VR-5, never upgraded).** The
        // op's *definition* is "the correctly-rounded IEEE-754 binary64 result under RNE" (`Exact`
        // as a definition — it is the spec); the *implementation claim* that the host's f64
        // arithmetic delivers exactly that bit pattern is **`Empirical`** at introduction (pinned
        // by the hand-derived IEEE reference-case corpus in `mycelium-interp`), with the
        // underlying "Rust f64 is IEEE-754 binary64" platform statement held at `Declared` (the
        // Rust reference; not independently verified). No `Proven` is claimed anywhere: a Proven
        // accuracy-vs-real-arithmetic claim would need a theorem with *checked* side-conditions
        // (none is checked here), and — unlike `bin.*`, whose two's-complement kernel is
        // in-project, decidable software — these ops delegate to host float hardware, so `Exact`
        // would overstate the conformance evidence (the ADR's own tag table). libm is NOT
        // involved (ADR-040 §2.5 keeps transcendentals out of the kernel), so this is not the
        // Empirical-libm case — the Empirical here is the host-conformance claim, disclosed as a
        // zero-deviation-vs-spec bound on the runtime result (see `mycelium-interp`'s wrappers).
        //
        // **Paradigm/width-model note (FLAG — the same escape hatch as the seq/bytes/dense prims
        // above):** `PrimParadigm` has no first-class `Float` paradigm, so operands/result are
        // typed `Any`/`Uniform` here as the nearest tags. The real never-silent typing — every
        // operand a `Float` (binary64) scalar — is enforced by the interpreter prims
        // (`prims.rs::as_float`) and the L1 checker (`checkty.rs::try_check_float_prim`). A
        // first-class `Float` paradigm is a deliberate, append-only extension left for the
        // surface-typing work (it ripples into content-addressing), exactly as for `Dense`.
        let flt = |operands: Vec<PrimParadigm>| PrimDecl {
            sig: PrimSig {
                operands,
                result: Any,
                width: WidthRel::Uniform,
            },
            intrinsic: GuaranteeStrength::Empirical,
        };
        t.insert("flt.add", flt(vec![Any, Any]));
        t.insert("flt.sub", flt(vec![Any, Any]));
        t.insert("flt.mul", flt(vec![Any, Any]));
        t.insert("flt.div", flt(vec![Any, Any]));
        t.insert("flt.neg", flt(vec![Any]));
        // ADR-040 §2.4 (M-899, `enb` Gap A): the **scalar-float comparison group** — two `Float`
        // operands reduce to a `Binary{1}` truth value (`WidthRel::Collapse`, the `cmp.eq`/
        // `cmp.lt` shape; the realized `Bool` of RFC-0032 D1's engineering note).
        //
        // **Explicit NaN semantics — the ADR-040 §2.4 partial order.** `flt.lt`/`flt.le`/
        // `flt.gt`/`flt.ge`/`flt.eq` are the IEEE-754 §5.11 quiet comparison *predicates*: float
        // ordering is **partial**, and a comparison involving NaN is the *defined* predicate
        // value **false** on every predicate (`flt.eq(NaN, NaN) = false` — NaN ≠ NaN). "False"
        // from `flt.lt` asserts "no `<` relation holds", never "≥": the no-order case is
        // explicitly *observable* from the predicate set itself (`¬le(a,b) ∧ ¬gt(a,b)` ⟺
        // unordered; `¬eq(x,x)` ⟺ NaN), so nothing is silently funneled into an ordering (G2 —
        // the §2.4 "never a silent false-as-less-than" clause; the `Option`-shaped three-way
        // `partial_cmp` is the `std.cmp` surface built *on* these predicates, cmp.md Q1, not a
        // kernel prim). **`flt.total_le` is the named, opt-in total order** — IEEE-754 §5.10
        // `totalOrder(a, b)` (a precedes-or-equals b): `−inf < … < −0 < +0 < … < +inf < NaN`
        // (the canonical positive quiet NaN of §2.3 sorts *last*, and `total_le` is reflexive on
        // NaN where `flt.le` is not; `−0`/`+0` are *distinct* under it where `flt.eq` calls them
        // equal — the FLAG-4 identity-vs-equality seam, made orderable *by name*, never
        // silently). Sorting/keying routes through `flt.total_le` explicitly — imposing a total
        // order silently is exactly what cmp.md Q1 rejects.
        //
        // **Intrinsic — `Empirical`, per the ratified ADR-040 §2.6 (VR-5, never upgraded):**
        // partial-order behavior is `Empirical` (property-tested, NaN cases in conformance —
        // the host-`f64`-operators-implement-IEEE-§5.11 claim rests on the `Declared` Rust
        // platform statement, pinned by the reference corpus in `mycelium-interp`), and the
        // `totalOrder` total-order *property* (totality/antisymmetry/transitivity) **stays
        // `Empirical` until a proof lands — the M-511 proof debt, load-bearing here and NOT
        // claimed `Proven`** (no checked side-condition theorem exists yet).
        //
        // Paradigm note: operands are the same documented `Any` escape hatch as the arithmetic
        // group above (no first-class `Float` paradigm yet); the result genuinely IS `Binary{1}`,
        // so `result: Binary` is precise, not a hatch.
        let flt_cmp = || PrimDecl {
            sig: PrimSig {
                operands: vec![Any, Any],
                result: Binary,
                width: WidthRel::Collapse,
            },
            intrinsic: GuaranteeStrength::Empirical,
        };
        t.insert("flt.lt", flt_cmp());
        t.insert("flt.le", flt_cmp());
        t.insert("flt.gt", flt_cmp());
        t.insert("flt.ge", flt_cmp());
        t.insert("flt.eq", flt_cmp());
        t.insert("flt.total_le", flt_cmp());
        // ADR-040 §2.5 (CU-2): the mandated float classification predicates — unary `Float →
        // Binary{1}` (the direct never-silent tests for the in-band ±inf/NaN sentinels, §2.4).
        // Same `Any` operand escape hatch as the comparison group (no first-class `Float` paradigm
        // in the sig yet); the result genuinely IS `Binary{1}`. Tag `Empirical` (ADR-040 §2.6).
        let flt_class = || PrimDecl {
            sig: PrimSig {
                operands: vec![Any],
                result: Binary,
                width: WidthRel::Collapse,
            },
            intrinsic: GuaranteeStrength::Empirical,
        };
        t.insert("flt.is_nan", flt_class());
        t.insert("flt.is_finite", flt_class());
        t.insert("flt.is_infinite", flt_class());
        // ADR-040 §2.4 (CU-3): never-silent Binary↔Float conversions — the "target-width prim"
        // shape of `bit.width_cast` (DN-41), crossing the Binary/Float paradigms. `bin.to_flt`
        // is **checked-exact** (refuses when the Binary operand's unsigned magnitude exceeds
        // `2^53`, binary64's exact-integer bound); `flt.to_bin` refuses on NaN/±inf/negative/
        // fractional/out-of-target-width, mirroring `bit.width_cast`'s witness-operand shape
        // (`value: Float, into: Binary{M}) -> Binary{M}`, `M` read from the second operand's
        // width only). The **lossy** rounding `flt(bin(n))` direction for `|n| > 2^53` is
        // explicitly out of scope — a reified *swap* carrying its bound (ADR-040 §2.4/§5), not a
        // prim (see the CU-3 leaf report FLAG).
        //
        // **Paradigm/width-model note (FLAG — the same escape hatch as the `flt.*` group above):**
        // `PrimParadigm` has no first-class `Float` paradigm, so both operands/results are typed
        // `Any` here; the real never-silent typing (the unsigned-magnitude reading, the checked-
        // exact `2^53` bound, `flt.to_bin`'s width-witness) is enforced by the interpreter prims
        // (`prims.rs::{prim_bin_to_flt,prim_flt_to_bin}`) and the L1 checker
        // (`checkty.rs::try_check_float_prim`). Width `Uniform` is the nearest tag (no
        // first-class width-*change* relation exists yet — the same width-model FLAG as
        // `bit.width_cast`); `flt.to_bin`'s real result width is its witness operand's width.
        //
        // **Intrinsic `Empirical` (ADR-040 §2.6 — "Conversions: range/exactness checks Empirical
        // via property tests on the documented bounds (2^53, target-range edges)"), NOT `Exact`.**
        // Unlike `bit.width_cast` (a pure bit-pattern re-width with no float involved, hence
        // `Exact`), both conversions here cross into `Repr::Float` territory and inherit the same
        // ADR-040 §2.6 host-conformance posture the `flt.*` arithmetic/comparison groups carry —
        // pinned by the same `empirical_flt_result` composition path (`mycelium-interp`).
        let empirical = |operands: Vec<PrimParadigm>, result: PrimParadigm| PrimDecl {
            sig: PrimSig {
                operands,
                result,
                width: WidthRel::Uniform,
            },
            intrinsic: GuaranteeStrength::Empirical,
        };
        t.insert("bin.to_flt", empirical(vec![Any], Any));
        t.insert("flt.to_bin", empirical(vec![Any, Any], Any));
        // RFC-0003 §3/§4 / ADR-008 (M-892, `enb` Gap C): the **VSA bind group** —
        // `vsa.bind`/`vsa.unbind`/`vsa.permute` over `Repr::Vsa{model, dim, sparsity}` values,
        // **model-dispatched** at runtime on the operand's model id across the introduction set
        // **MAP-I / FHRR / BSC** (an operand outside that set is an explicit refusal in the
        // interpreter wrapper, never a guessed algebra — G2; widening the set is an append-only
        // extension that must recompute the meets below). The kernel (`mycelium-vsa`'s Value-level
        // ops, e.g. `MapI::bind_values`) constructs the full result `Value` — payload and `Meta`
        // (model-namespaced `Derived` provenance such as `vsa.map_i.bind`, and the per-model
        // honest tag) — and the wrapper carries it through unchanged (VR-5), exactly the M-890/
        // M-891 tensor-valued pattern.
        //
        // **Intrinsic tags — the MEET over the dispatch set, never the strongest member (VR-5).**
        // A Π declaration stores ONE strength, but the per-op tag is *per-model* (RFC-0003 §4:
        // MAP-I bind/unbind/permute `Exact`; FHRR bind/permute `Exact` but unbind **`Empirical`**
        // — the normative weak-link assignment; BSC bind/unbind/permute `Exact`). Recording the
        // strongest would over-claim for FHRR, so the table records the meet: `vsa.bind`/
        // `vsa.permute` = `Exact` (all three agree), `vsa.unbind` = **`Empirical`** (downgraded to
        // stay accurate — house rule 1). The *runtime* result still carries the dispatched model's
        // own (possibly stronger) kernel tag — e.g. a MAP-I unbind result is `Exact` — because the
        // kernel constructs the `Meta`, not this table. Table↔kernel meet-consistency is guarded
        // by a `mycelium-interp` test (this crate cannot see `mycelium-vsa` — ADR-008 keeps the
        // dependency one-way).
        //
        // **Paradigm/width-model note (FLAG — the same escape hatch as the seq/bytes/dense/flt
        // prims above):** `PrimParadigm` has no first-class `Vsa` paradigm and `WidthRel` no
        // model/dim relation, so operands/results are typed `Any`/`Uniform` here as the nearest
        // tags (`vsa.permute`'s second operand is really a `Binary{W}` shift amount — enforced by
        // the interpreter wrapper and the L1 checker, like `bit.width_cast`'s witness operand).
        // The real never-silent typing — equal model + dim on every hypervector operand, model
        // mismatch an explicit refusal, never a coercion — is enforced by the kernel + interpreter
        // prim and the L1 checker branch (`checkty.rs::try_check_vsa_prim`). A first-class `Vsa`
        // paradigm is a deliberate, RFC-unpinned extension left for the surface-typing work
        // (it ripples into content-addressing), exactly as for `Seq`/`Bytes`/`Dense`/`Float`.
        let vsa = |operands: Vec<PrimParadigm>, intrinsic: GuaranteeStrength| PrimDecl {
            sig: PrimSig {
                operands,
                result: Any,
                width: WidthRel::Uniform,
            },
            intrinsic,
        };
        t.insert("vsa.bind", vsa(vec![Any, Any], GuaranteeStrength::Exact));
        t.insert(
            "vsa.unbind",
            vsa(vec![Any, Any], GuaranteeStrength::Empirical),
        );
        t.insert("vsa.permute", vsa(vec![Any, Any], GuaranteeStrength::Exact));
        // RFC-0003 §4/§5 / ADR-008 (M-893, `enb` Gap C): **`vsa.bundle`** — superposition via the
        // **certified path** (`MapI::bundle_values_certified` in `mycelium-vsa`). Operands are a
        // `Seq` of hypervectors and a `Float` target failure probability δ (both typed `Any` under
        // the same paradigm-model escape hatch as the bind group above; the real typing —
        // `Seq{Vsa{m, d}, N≥1}` × `Float` → `Vsa{m, d}` — is enforced by the interpreter prim and
        // the L1 checker branch).
        //
        // **The dispatch set for bundle is the certified singleton {MAP-I}** — the only
        // introduction-set model with a *certified* Value-level bundle (the M-131
        // checked-instantiation pattern: a `Proven` `CapacityBound` citing Clarkson/Thomas is
        // issued **iff** `dim ≥ requiredDim(m, δ)` is checked, with bipolar + distinct items also
        // checked; otherwise an explicit refusal, never an unbacked tag). FHRR/BSC bundles are
        // **`Empirical`-profile ops** in the kernel — routing them through this prim would either
        // silently downgrade the prim's meaning or silently upgrade their tag (both VR-5
        // violations), so they are explicit refusals in the wrapper/checker; surfacing them is a
        // distinct, append-only extension under its own name. The intrinsic is therefore the meet
        // over that certified singleton = MAP-I's `Bundle` tag = **`Proven`**; the runtime value
        // carries the kernel-checked `CapacityBound` itself (kernel↔table consistency is guarded
        // by a `mycelium-interp` test — this crate cannot see `mycelium-vsa`).
        t.insert("vsa.bundle", vsa(vec![Any, Any], GuaranteeStrength::Proven));
        // RFC-0003 §3/§6 / ADR-008 (M-894, `enb` Gap C): **`vsa.cleanup`** + **`vsa.reconstruct`**
        // — the cleanup-memory retrieval and the compositional role-reconstruction decode (FR-S4),
        // plus **`vsa.required_dim`**, the capacity-bound query (RFC-0003 §5; M-131).
        //
        // `vsa.cleanup(query, codebook)` snaps a (possibly noisy) hypervector to the nearest
        // codebook atom by the dispatched model's similarity and returns the **decision triple**
        // `[index, confidence, margin]` (a `Seq{Float, 3}`) — the retrieval is never a silent
        // nearest-neighbour pick (FR-S4/G2: confidence + margin are reported in-band, the caller
        // decides). The decision procedure is an exhaustive arg-max over the codebook guarded by
        // the RFC-0010 §4.4 identifiability refusal (a tie is an explicit error, never a
        // coin-flip), so the intrinsic is **`Exact`** — the same claim shape as the RFC-0010
        // brute-force decode arm — uniformly across the MAP-I/FHRR/BSC dispatch set (the model
        // only supplies `similarity`; the procedure is model-generic, so the meet is `Exact`).
        // A non-`Exact` **query** does not refuse: its (strength, bound) pair passes through to
        // the result via the RFC-0001 §4.7 meet (the M-204 `Passthrough` posture — cleanup exists
        // precisely to make a noisy unbind usable), while codebook atoms must be `Exact`.
        //
        // `vsa.reconstruct(record, role, codebook, threshold)` is the RFC-0003 §6 compositional
        // reconstruction (`reconstruct_role` semantics): unbind the record by the role atom, clean
        // the noisy result up against the codebook, and **refuse explicitly below the caller's
        // `Float` threshold** (the manifest's `cleanup_threshold` made an explicit operand —
        // never a silent low-quality answer, G2). Result: the same `Seq{Float, 3}` decision
        // triple; the record's own (strength, bound) pair passes through (a certified bundle's
        // `Proven` `CapacityBound` is re-disclosed on the decode — the disclosed bound is the
        // value's own). **The dispatch set for reconstruct is {MAP-I, BSC}** — the models whose
        // unbind is `Exact` self-inverse algebra; FHRR's unbind tag is `Empirical` and
        // trial-validated only for a single `vsa.fhrr.bind` product (the kernel's regime gate),
        // which a reconstruction record is not, so an FHRR reconstruct is an explicit refusal
        // (never a stretched profile — VR-5); surfacing it is an append-only extension under a
        // reconstruction-regime profile of its own. The intrinsic is the meet over {MAP-I, BSC}
        // of unbind∘arg-max = **`Exact`**. The factor-decode sibling (`reconstruct_factors`,
        // RFC-0009/RFC-0010) is deliberately NOT surfaced here: it routes through the RFC-0005
        // selector whose mandatory EXPLAIN has no prim-surface carrier yet, and its manifest/
        // multi-codebook forms need value shapes this surface lacks — a distinct, append-only
        // surfacing under its own name (`vsa.reconstruct_factors`), never a silent conflation.
        //
        // `vsa.required_dim(items, δ)` surfaces the M-131 capacity-bound query: the sufficient
        // dimension `requiredDim(m, δ) = ⌈(2/μ²)·ln(m/δ)⌉` (μ = 0.1, the cited Clarkson/Thomas
        // instantiation — `mycelium-vsa::capacity`). The result is a `Binary{64}` dimension
        // carrying the kernel's **`Proven`** `CapacityBound` for exactly that (items, dim, δ)
        // instantiation (`proven_capacity_bound` — the side-condition `dim ≥ requiredDim` holds
        // by construction), so the query is inspectable/EXPLAIN-able: the `ProvenThm` basis
        // records the citation, μ, and the checked condition. Intrinsic **`Proven`** — the same
        // checked-instantiation stance as `vsa.bundle`. Degenerate inputs (zero items, δ outside
        // `(0, 1]`) are explicit wrapper refusals, never the kernel's `u64::MAX` sentinel.
        //
        // All three ride the same `Any`/`Uniform` paradigm-model escape hatch as the bind group
        // (the real typing — `Vsa{m, d}` × `Seq{Vsa{m, d}, N≥1}` → `Seq{Float, 3}`,
        // `Vsa{m, d}` × `Vsa{m, d}` × `Seq{Vsa{m, d}, N≥1}` × `Float` → `Seq{Float, 3}`,
        // `Binary{W}` × `Float` → `Binary{64}` — is enforced by the interpreter prim + the L1
        // checker branch `try_check_vsa_prim`; table↔kernel consistency is guarded by a
        // `mycelium-interp` test, ADR-008 keeping the dependency one-way).
        t.insert("vsa.cleanup", vsa(vec![Any, Any], GuaranteeStrength::Exact));
        t.insert(
            "vsa.reconstruct",
            vsa(vec![Any, Any, Any, Any], GuaranteeStrength::Exact),
        );
        t.insert(
            "vsa.required_dim",
            vsa(vec![Any, Any], GuaranteeStrength::Proven),
        );
        t
    }

    /// The content hash of the prim registered under kernel name `name`, if any.
    #[must_use]
    pub fn decl_hash(&self, name: &str) -> Option<&ContentHash> {
        self.by_name.get(name)
    }

    /// A [`PrimRef`] for the prim named `name`, if registered.
    #[must_use]
    pub fn prim_ref(&self, name: &str) -> Option<PrimRef> {
        self.by_name.get(name).cloned().map(PrimRef::new)
    }

    /// The resolved declaration at content hash `hash`, if registered.
    #[must_use]
    pub fn decl(&self, hash: &ContentHash) -> Option<&PrimDecl> {
        self.decls.get(hash)
    }

    /// The declaration a [`PrimRef`] points at, if registered.
    #[must_use]
    pub fn resolve(&self, prim: &PrimRef) -> Option<&PrimDecl> {
        self.decls.get(prim.decl())
    }

    /// The declaration registered under kernel name `name`, if any.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PrimDecl> {
        let hash = self.by_name.get(name)?;
        self.decls.get(hash)
    }

    /// The intrinsic guarantee `g_f` of the prim named `name` (RFC-0001 §4.7), if registered.
    #[must_use]
    pub fn intrinsic(&self, name: &str) -> Option<GuaranteeStrength> {
        self.get(name).map(|d| d.intrinsic)
    }

    /// Whether a prim named `name` is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// The registered kernel names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.by_name.keys().map(String::as_str).collect()
    }

    /// Every entry as `(name, #p, decl)`, in name order — the inspectable surface for EXPLAIN over
    /// prims (DN-10 §3.2 step 4; G2/SC-3): a prim call can report which content-addressed
    /// declaration it resolves to, its signature, and its intrinsic guarantee.
    #[must_use]
    pub fn entries(&self) -> Vec<(&str, PrimRef, &PrimDecl)> {
        self.by_name
            .iter()
            .filter_map(|(name, hash)| {
                self.decls
                    .get(hash)
                    .map(|d| (name.as_str(), PrimRef::new(hash.clone()), d))
            })
            .collect()
    }
}
