//! The **M-961 inject-mode conformance suite** (RFC-0038 §13; DN-77 §4) — parameterized over
//! `InjectMode`, covering the §13 clauses **for the built subset**:
//!
//! * (a) `loose` permits unsigned injection with G2 tagging — [`loose_admits_unsigned_with_g2_tag`]
//! * (b) `inoculated` rejects unsigned code with `UnsignedCode(ContentHash)` — the refusal table
//! * (c) `inoculated` rejects unsigned **interpreted** definitions (I6) — the refusal table
//! * (d) `TrustRoot` immutability — runtime change yields an explicit error
//! * (e) `inject_mode` appears on every `Resolution` and is EXPLAIN-able
//! * (f) cert axis × inject axis compose in all four combinations (I5)
//! * (g) `whole` grain honored; `module`/`call` refuse never-silently (M-847); the deviation
//!   manifest renders the declared default plus the enumerated departures
//! * (h) `BadSignature` rejects a wrong/untrusted signer on **both** paths. *(The blacklist half
//!   of (h) is §8.8 colony-topology scope — deferred with M-849, stays `Declared`; recorded here
//!   explicitly, never silently skipped — G2.)*
//!
//! Plus the M-961 DoD **three-way differential**: reference-interp ≡ loose-unsigned ≡
//! inoculated-signed, on the interpreted path always, and on the compiled path **where it is
//! executable** — a missing `clang` is an explicit, printed skip (the house idiom), never a
//! silent one.
//!
//! Guarantee tags per test doc; the suite exercises the **gate mechanism** (Enacted with this
//! code); the test scheme's unforgeability is NOT claimed (`Declared` — M-836).

use crate::inject_gate::{
    Admission, EnforcementGrain, InjectMode, InjectPolicy, PolicyError, SignerId, TestScheme,
    TrustRoot,
};
use mycelium_core::{ContentHash, Meta, Payload, Provenance, Repr, Value};
use mycelium_interp::Interpreter;

use crate::inject::{Image, InjectError, Resolution};
use crate::inject_cert::InjectCert;
use crate::llvm::AotError;
use crate::vr4::cross_backend_gate;

// ─── fixtures ─────────────────────────────────────────────────────────────────

const ISSUED_AT: u64 = 1_751_400_000; // §L placeholder — carried, not enforced (M-837).

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `not(<bits>)` — a closed bit-subset program (the JIT's domain).
fn not_prog(bits: Vec<bool>) -> mycelium_core::Node {
    mycelium_core::Node::Op {
        prim: "bit.not".into(),
        args: vec![mycelium_core::Node::Const(binary(bits))],
    }
}

fn signer(s: &str) -> SignerId {
    SignerId::new(s)
}

fn root(names: &[&str]) -> TrustRoot {
    TrustRoot::new(names.iter().map(|n| signer(n)))
}

/// An inoculated/whole image trusting `names` (the RFC-0038 §8.6 application default).
fn inoculated_image(names: &[&str]) -> Image {
    let policy = InjectPolicy::germinate(
        InjectMode::Inoculated,
        EnforcementGrain::Whole,
        root(names),
        Box::new(TestScheme),
    )
    .expect("inoculated/whole with a non-empty root germinates");
    Image::germinate(Interpreter::default(), policy)
}

/// A loose image with a non-empty root — the §8.7 per-inject signing opt-in context.
fn loose_image_with_root(names: &[&str]) -> Image {
    let policy = InjectPolicy::germinate(
        InjectMode::Loose,
        EnforcementGrain::Whole,
        root(names),
        Box::new(TestScheme),
    )
    .expect("loose germinates with any root");
    Image::germinate(Interpreter::default(), policy)
}

/// A valid test-scheme cert for `node` from `who` (the dev/test issuing path — production
/// emission is M-839/M-836, `Declared`).
fn cert_for(node: &mycelium_core::Node, who: &str) -> InjectCert {
    InjectCert::issue_with_test_scheme(
        signer(who),
        node.content_hash(),
        cross_backend_gate(node),
        ISSUED_AT,
    )
}

fn toolchain_missing(e: &InjectError) -> bool {
    matches!(e, InjectError::Compile(AotError::ToolchainMissing(_)))
}

// ─── (b)/(c): the never-silent unsigned refusals, both paths (I2/I6) ──────────

/// **§13 (b)+(c):** an `inoculated` image refuses *unsigned* registration on BOTH paths — the
/// compiled (`inject`) and the interpreter fallback (`define`) — with the explicit
/// `UnsignedCode(ContentHash)` carrying the exact rejected hash, and registers NOTHING (never a
/// partial admission). Guarantee: `Proven` for the refusal firing (direct construction).
#[test]
fn inoculated_refuses_unsigned_on_both_paths() {
    let prog = not_prog(vec![true, false, true, true]);
    let expected_hash = prog.content_hash();

    // Path 1 — interpreter fallback (I6): define.
    let mut img = inoculated_image(&["k1"]);
    assert_eq!(
        img.define(prog.clone()),
        Err(InjectError::UnsignedCode(expected_hash.clone())),
        "unsigned define must refuse on an inoculated image (I6)"
    );
    // Path 2 — compiled: inject.
    assert_eq!(
        img.inject(&prog),
        Err(InjectError::UnsignedCode(expected_hash.clone())),
        "unsigned inject must refuse on an inoculated image (I2)"
    );
    // Nothing registered on either refusal.
    assert_eq!(img.defined_count(), 0, "no partial registration");
    assert_eq!(img.injected_count(), 0, "no partial registration");
    assert_eq!(img.resolve(&expected_hash), Resolution::Miss);
    // The refusal explains itself (G2).
    let msg = format!("{}", InjectError::UnsignedCode(expected_hash));
    assert!(
        msg.contains("inoculated") && msg.contains("RFC-0038"),
        "{msg}"
    );
}

/// **§13 (h), built half:** a cert from an untrusted signer, a tampered signature, and a cert
/// minted for *different content* are each an explicit `BadSignature(ContentHash, SignerId)` on
/// BOTH paths — and a presented-but-bad cert is blocked **even in `loose` mode** (§8.7: bad,
/// untrusted code is never downgraded to unsigned-permitted). Nothing is registered.
/// *(The §8.8 blacklist half of (h) is deferred with M-849 — `Declared`, recorded, not tested
/// here because the surface does not exist; G2.)*
/// Guarantee: `Proven` for the refusal firing.
#[test]
fn bad_signature_refuses_on_both_paths_and_in_both_modes() {
    let prog = not_prog(vec![false, true, false, true]);
    let other = not_prog(vec![true, true, true, true]);
    let hash = prog.content_hash();

    // The refusal case table: (case name, cert, claimed signer) — each must refuse as BadSignature.
    let untrusted = cert_for(&prog, "mallory");
    let wrong_content = cert_for(&other, "k1");
    let tampered = {
        // A valid cert for `prog` from the trusted signer, with one signature byte flipped.
        let good = cert_for(&prog, "k1");
        let mut sig = good.signature().to_vec();
        sig[0] = sig[0].wrapping_add(1);
        InjectCert::from_parts(
            signer("k1"),
            prog.content_hash(),
            sig,
            cross_backend_gate(&prog),
            ISSUED_AT,
        )
    };

    let cases: Vec<(&str, InjectCert, SignerId)> = vec![
        ("untrusted signer", untrusted, signer("mallory")),
        ("cert for different content", wrong_content, signer("k1")),
        ("tampered/mismatched signature", tampered, signer("k1")),
    ];

    for (name, cert, who) in &cases {
        // Inoculated image, both paths.
        let mut ino = inoculated_image(&["k1"]);
        assert_eq!(
            ino.define_signed(prog.clone(), cert),
            Err(InjectError::BadSignature(hash.clone(), who.clone())),
            "inoculated define_signed: {name}"
        );
        assert_eq!(
            ino.inject_signed(&prog, cert),
            Err(InjectError::BadSignature(hash.clone(), who.clone())),
            "inoculated inject_signed: {name}"
        );
        assert_eq!(ino.defined_count(), 0, "{name}: no partial registration");
        assert_eq!(ino.injected_count(), 0, "{name}: no partial registration");

        // Loose image with the same root: presented-but-bad certs are BLOCKED, not downgraded.
        let mut loose = loose_image_with_root(&["k1"]);
        assert_eq!(
            loose.define_signed(prog.clone(), cert),
            Err(InjectError::BadSignature(hash.clone(), who.clone())),
            "loose define_signed must block a bad cert (§8.7): {name}"
        );
        assert_eq!(loose.defined_count(), 0, "{name}: nothing registered");
    }
}

// ─── (a)/(e): loose admission is G2-tagged; inject_mode on every Resolution ───

/// **§13 (a)+(e):** `loose` admits unsigned registration, and the admission is **G2-tagged** on
/// the `Resolution` (I1) — `inject_mode` + `Unsigned` admission appear on the interpreted entry
/// and are EXPLAIN-able via the manifest. Guarantee: `Proven` (direct construction).
#[test]
fn loose_admits_unsigned_with_g2_tag() {
    let mut img = Image::new(); // loose dev default: empty TrustRoot ⇒ loose (§7.1, explicit).
    assert_eq!(img.inject_mode(), InjectMode::Loose);
    let prog = not_prog(vec![true, false]);
    let hash = img.define(prog).expect("loose admits unsigned");
    assert_eq!(
        img.resolve(&hash),
        Resolution::Interpreted {
            inject_mode: InjectMode::Loose,
            admission: Admission::Unsigned,
        },
        "the unsigned status is tagged on the dispatch decision, never silent (I1/I7)"
    );
    // EXPLAIN surfaces the posture (e) — and the scheme's honesty label (VR-5).
    let explain = img.explain_policy();
    assert!(explain.contains("loose"), "{explain}");
    assert!(explain.contains("NOT production crypto"), "{explain}");
}

/// **Signed admission on the interpreter path (always executable):** an `inoculated` image
/// admits a definition with a valid cert from a trusted signer; the `Resolution` carries the
/// `Verified` admission with the signer (I7). Guarantee: `Proven` (direct construction; the
/// scheme's binding is `Empirical` per the mycelium-sec property tests).
#[test]
fn inoculated_admits_verified_define_and_tags_the_signer() {
    let prog = not_prog(vec![true, true, false, true]);
    let mut img = inoculated_image(&["k1"]);
    let hash = img
        .define_signed(
            prog,
            &cert_for(&not_prog(vec![true, true, false, true]), "k1"),
        )
        .expect("a valid cert from a trusted signer admits");
    assert_eq!(
        img.resolve(&hash),
        Resolution::Interpreted {
            inject_mode: InjectMode::Inoculated,
            admission: Admission::Verified {
                signer: signer("k1")
            },
        }
    );
    let v = img.call(&hash).expect("admitted definition is callable");
    assert_eq!(v.payload(), &Payload::Bits(vec![false, false, true, false]));
}

// ─── (d): TrustRoot immutability ──────────────────────────────────────────────

/// **§13 (d):** the `TrustRoot` is immutable after germination — a runtime change attempt is an
/// explicit `TrustRootImmutable` error (I3), and the root is observably unchanged.
/// Guarantee: `Proven` (the refusing method is total; there is no other mutation API).
#[test]
fn trust_root_is_immutable_after_germination() {
    let mut img = inoculated_image(&["k1"]);
    let before = img.trust_root().clone();
    assert_eq!(
        img.set_trust_root(root(&["k1", "mallory"])),
        Err(InjectError::TrustRootImmutable)
    );
    assert_eq!(img.trust_root(), &before, "the root is unchanged");
    // And germination itself refuses the incoherent posture (inoculated + empty root) rather
    // than silently downgrading (§7.1/G2).
    assert_eq!(
        InjectPolicy::germinate(
            InjectMode::Inoculated,
            EnforcementGrain::Whole,
            TrustRoot::empty(),
            Box::new(TestScheme),
        )
        .err(),
        Some(PolicyError::EmptyTrustRoot)
    );
}

// ─── (f): the cert axis and the inject axis compose (I5) ──────────────────────

/// **§13 (f):** the cert axis (RFC-0034 `CertMode`, a `Meta` field) and the inject axis compose
/// in all 2×3 combinations: the gate decision depends only on the inject mode, and the dispatch
/// key is identical across cert modes (RFC-0034 §8 — `Meta` is excluded from the content hash).
/// Guarantee: `Proven` by construction (RFC-0001 §4.6) + exhaustive over the combinations.
#[test]
fn cert_axis_and_inject_axis_compose_freely() {
    use mycelium_core::CertMode;
    let mut keys: Vec<ContentHash> = Vec::new();
    for cert_mode in &CertMode::ALL {
        // Embed the cert-mode tag in the program's value Meta (the cert axis).
        let val = Value::new(
            Repr::Binary { width: 2 },
            Payload::Bits(vec![true, false]),
            Meta::exact(Provenance::Root).with_cert_mode(*cert_mode),
        )
        .unwrap();
        let prog = mycelium_core::Node::Op {
            prim: "bit.not".into(),
            args: vec![mycelium_core::Node::Const(val)],
        };
        let expected_hash = prog.content_hash();

        // Inject axis: loose admits unsigned …
        let mut loose = Image::new();
        let h = loose.define(prog.clone()).expect("loose admits unsigned");
        keys.push(h.clone());
        // … inoculated refuses the SAME program unsigned (the gate ignores the cert axis) …
        let mut ino = inoculated_image(&["k1"]);
        assert_eq!(
            ino.define(prog.clone()),
            Err(InjectError::UnsignedCode(expected_hash)),
            "inoculated refuses unsigned regardless of CertMode (I5)"
        );
        // … and admits it signed.
        ino.define_signed(prog.clone(), &cert_for(&prog, "k1"))
            .expect("inoculated admits a valid cert regardless of CertMode (I5)");
    }
    // The dispatch key never varies with the cert axis (RFC-0034 §8).
    assert!(keys.windows(2).all(|w| w[0] == w[1]), "{keys:?}");
}

// ─── (g): whole grain + never-silent unbuilt grains + the deviation manifest ──

/// **§13 (g), scoped per DN-77 §4 item 6:** the `whole` grain is the enforced Phase-I default;
/// selecting `module`/`call` refuses never-silently at germination (their enforcement paths are
/// M-847's tracked scope — never a silent downgrade to `whole`); and the **deviation manifest**
/// renders the declared default plus the enumerated departures (§8.5).
/// Guarantee: `Proven` for the refusals/rendering.
#[test]
fn whole_grain_manifest_and_unbuilt_grain_refusals() {
    // Unbuilt grains refuse explicitly (the DN-63 pattern), citing the owning issue.
    for grain in [EnforcementGrain::Module, EnforcementGrain::Call] {
        let got = InjectPolicy::germinate(
            InjectMode::Inoculated,
            EnforcementGrain::Whole,
            root(&["k1"]),
            Box::new(TestScheme),
        )
        .and_then(|_| {
            InjectPolicy::germinate(
                InjectMode::Inoculated,
                grain,
                root(&["k1"]),
                Box::new(TestScheme),
            )
        });
        assert_eq!(got.err(), Some(PolicyError::GrainNotYetEnforced(grain)));
    }

    // The manifest: a loose image whose one signed admission is an enumerated deviation (§8.7).
    let prog = not_prog(vec![true, false, false]);
    let mut img = loose_image_with_root(&["k1"]);
    let unsigned = not_prog(vec![false, false, false]);
    img.define(unsigned).expect("unsigned is the loose norm");
    let hash = img
        .define_signed(prog.clone(), &cert_for(&prog, "k1"))
        .expect("per-inject signing opt-in admits");

    let manifest = img.policy_manifest();
    assert_eq!(manifest.mode, InjectMode::Loose);
    assert_eq!(manifest.grain, EnforcementGrain::Whole);
    assert_eq!(manifest.deviations.len(), 1, "exactly the signed site");
    assert_eq!(manifest.deviations[0].site, hash.as_str());
    let explain = manifest.explain();
    assert!(explain.contains("deviations (1):"), "{explain}");
    assert!(explain.contains("§8.7"), "{explain}");

    // An inoculated image's manifest: verified is the norm — no deviations, said explicitly.
    let mut ino = inoculated_image(&["k1"]);
    ino.define_signed(prog.clone(), &cert_for(&prog, "k1"))
        .expect("admits");
    let e2 = ino.explain_policy();
    assert!(e2.contains("deviations: none"), "{e2}");
    assert!(e2.contains("inoculated"), "{e2}");
}

// ─── the M-961 DoD three-way differential ─────────────────────────────────────

/// **Three-way differential (M-961 DoD):** the same program yields the same observable through
/// (1) the reference interpreter, (2) a loose image's unsigned path, and (3) an inoculated
/// image's signed path — on the **interpreted** leg always, and on the **compiled** leg where it
/// is executable: a missing `clang` toolchain is an **explicit, printed skip** of that leg only
/// (the house idiom — recorded, never silent; G2). Guarantee: `Empirical` (differential trials).
#[test]
fn three_way_differential_interp_loose_inoculated() {
    let prog = not_prog(vec![true, false, true, true]);
    let expected = Payload::Bits(vec![false, true, false, false]);

    // Leg 1 — the trusted reference interpreter (always executable).
    let reference = Interpreter::default()
        .eval(&prog)
        .expect("reference interpreter runs");
    assert_eq!(reference.payload(), &expected);

    // Leg 2 — loose image, unsigned interpreted path (always executable).
    let mut loose = Image::new();
    let h2 = loose.define(prog.clone()).expect("loose admits unsigned");
    let v2 = loose.call(&h2).expect("interpreted call runs");
    assert_eq!(v2.payload(), reference.payload(), "loose ≡ reference");

    // Leg 3 — inoculated image, signed interpreted path (always executable).
    let mut ino = inoculated_image(&["k1"]);
    let h3 = ino
        .define_signed(prog.clone(), &cert_for(&prog, "k1"))
        .expect("valid cert admits");
    let v3 = ino.call(&h3).expect("interpreted call runs");
    assert_eq!(v3.payload(), reference.payload(), "inoculated ≡ reference");

    // Compiled legs — where executable. A missing toolchain skips THIS LEG ONLY, explicitly.
    match loose.inject(&prog) {
        Ok(h) => {
            let v = loose.call(&h).expect("compiled call runs");
            assert_eq!(
                v.payload(),
                reference.payload(),
                "loose-compiled ≡ reference"
            );
        }
        Err(e) if toolchain_missing(&e) => {
            eprintln!(
                "SKIP (explicit, G2): loose compiled leg not executable here — {e}; \
                 the interpreted legs above still ran"
            );
        }
        Err(e) => panic!("unexpected inject error: {e}"),
    }
    match ino.inject_signed(&prog, &cert_for(&prog, "k1")) {
        Ok(h) => {
            let v = ino.call(&h).expect("compiled call runs");
            assert_eq!(
                v.payload(),
                reference.payload(),
                "inoculated-signed-compiled ≡ reference"
            );
            assert_eq!(
                ino.resolve(&h),
                Resolution::Compiled {
                    inject_mode: InjectMode::Inoculated,
                    admission: Admission::Verified {
                        signer: signer("k1")
                    },
                }
            );
        }
        Err(e) if toolchain_missing(&e) => {
            eprintln!(
                "SKIP (explicit, G2): inoculated compiled leg not executable here — {e}; \
                 the interpreted legs above still ran"
            );
        }
        Err(e) => panic!("unexpected inject error: {e}"),
    }
}
