//! M-740 Stage 4 (DN-26 §7.3 row 4) — the self-hosted `compiler.substrate` port.
//!
//! `lib/compiler/substrate.myc` ports `crates/mycelium-l1/src/substrate.rs`'s DETERMINISTIC
//! surface — provenance construction, identity, `explain`, the Live -> Consumed / Live ->
//! Released terminal transitions along ONE threaded value, and the error/event data. This is a
//! **unit differential** over small synthetic inputs (args-in/verdict-out, the
//! `compiler_stage2.rs`/`compiler_stage3.rs` one-eval economy pattern) — NOT a corpus sweep, since
//! `substrate.rs` is a runtime value-form module with no source-text grammar to sweep.
//!
//! **The central hazard (FLAG-substrate-1, full detail in `substrate.myc`'s own header comment):**
//! the Rust oracle's `consumed: Arc<AtomicBool>` is a flag SHARED across every `Clone` of one
//! identity — the runtime backstop for a closure/loop-body capture the static affine pass cannot
//! see. A pure-value port has no shared interior mutability (KC-3), so `substrate.myc`'s
//! `try_consume` can only enforce consume-once along a SINGLE threaded value, never across
//! aliases. This gate does **not** claim otherwise — it differentials the deterministic surface
//! only, on one handle value per case, never two aliases of the same identity.
//!
//! Other honest narrowings carried by `substrate.myc` itself (full detail in-file,
//! FLAG-substrate-2..6): `id` assignment is explicitly threaded (`acquire` takes/returns a
//! `next_id` counter) rather than a process-global atomic — this gate reads the REAL
//! oracle-assigned id back from a live `SubstrateHandle::acquire` call and feeds that exact value
//! to the `.myc` side, so both sides always agree on the SAME concrete identity (the differential
//! makes no claim about the Rust-side global counter's own numbering scheme); `id` is narrowed to
//! `Binary{32}` (cast down from the oracle's `u64`, always small in this gate); `explain` is
//! matched BYTE-FOR-BYTE against the live oracle (its format string is pure ASCII); `ReleaseEvent`/
//! `SubstrateError`'s `Display` prose is matched against a Rust-side MIRROR function using the
//! SAME ASCII-safe simplified template `substrate.myc` uses, not the live (non-ASCII) `Display`
//! impl — exactly the "hash/tag or classification code" allowance for string outputs.

use mycelium_l1::elab::build_registry;
use mycelium_l1::{
    check_nodule, monomorphize, parse, Evaluator, SubstrateError, SubstrateHandle,
    SubstrateProvenance,
};

const SUBSTRATE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/substrate.myc"
));

/// Format `n` as an explicit `Binary{32}` literal (bare decimal literals do not ambient-resolve
/// in every position — the `compiler_stage1.rs`/`compiler_stage2.rs` convention).
fn b32(n: u32) -> String {
    format!("0b{n:032b}")
}

fn escape_myc_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// The shared driver prelude: small classification-code helpers over `substrate.myc`'s public
/// surface, each unwrapping ONE constructor per `match` (split-match idiom, M-980). Lives in the
/// TEST file (not `substrate.myc` itself) — the `compiler_stage2.rs` convention: the ported
/// nodule stays a pure port, the differential-only driver code is appended here.
fn driver_prelude() -> String {
    format!(
        "fn sub_acquire_id_ok(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}) => Binary{{32}} =\n\
         \x20 match eq(sh_id(acquired_handle(acquire(tag, prov_new(av, site), id))), id) {{ 0b1 => {one}, _ => {zero} }};\n\
         fn sub_acquire_next_ok(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, want_next: Binary{{32}}) => Binary{{32}} =\n\
         \x20 match eq(acquired_next_id(acquire(tag, prov_new(av, site), id)), want_next) {{ 0b1 => {one}, _ => {zero} }};\n\
         fn sub_acquire_fresh_live(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}) => Binary{{32}} =\n\
         \x20 match sh_is_consumed(acquired_handle(acquire(tag, prov_new(av, site), id))) {{ False => {one}, True => {zero} }};\n\
         fn sub_explain_eq(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, want: Bytes) => Binary{{32}} =\n\
         \x20 match bytes_eq(explain(acquired_handle(acquire(tag, prov_new(av, site), id))), want) {{ 0b1 => {one}, _ => {zero} }};\n\
         fn sub_try_consume_first_ok(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match try_consume(h) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(h2) => match sh_is_consumed(h2) {{\n\
         \x20     False => {zero},\n\
         \x20     True => match eq(sh_id(h2), id) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn sub_try_consume_second_err(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match try_consume(h) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(h2) => match try_consume(h2) {{\n\
         \x20     Ok(_) => {zero},\n\
         \x20     Err(e) => match eq(se_id(e), id) {{\n\
         \x20       0b1 => match bytes_eq(se_tag(e), tag) {{ 0b1 => {one}, _ => {zero} }},\n\
         \x20       _ => {zero}\n\
         \x20     }}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn sub_se_display_eq(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, want: Bytes) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match try_consume(h) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(h2) => match try_consume(h2) {{\n\
         \x20     Ok(_) => {zero},\n\
         \x20     Err(e) => match bytes_eq(se_display(e), want) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn sub_release_fresh_some(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, rsite: Bytes) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match release(h, rsite) {{\n\
         \x20   None => {zero},\n\
         \x20   Some(re) => match eq(re_id(re), id) {{\n\
         \x20     0b1 => match bytes_eq(re_tag(re), tag) {{\n\
         \x20       0b1 => match bytes_eq(re_site(re), rsite) {{ 0b1 => {one}, _ => {zero} }},\n\
         \x20       _ => {zero}\n\
         \x20     }},\n\
         \x20     _ => {zero}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn sub_re_display_eq(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, rsite: Bytes, want: Bytes) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match release(h, rsite) {{\n\
         \x20   None => {zero},\n\
         \x20   Some(re) => match bytes_eq(re_display(re), want) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20 }};\n\
         fn sub_release_after_consume_none(tag: Bytes, av: Bytes, site: Bytes, id: Binary{{32}}, rsite: Bytes) => Binary{{32}} =\n\
         \x20 let h = acquired_handle(acquire(tag, prov_new(av, site), id)) in\n\
         \x20 match try_consume(h) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(h2) => match release(h2, rsite) {{ None => {one}, Some(_) => {zero} }}\n\
         \x20 }};\n\
         fn sub_handle_eq(tag1: Bytes, via1: Bytes, site1: Bytes, id1: Binary{{32}}, tag2: Bytes, via2: Bytes, site2: Bytes, id2: Binary{{32}}) => Binary{{32}} =\n\
         \x20 match sh_eq(acquired_handle(acquire(tag1, prov_new(via1, site1), id1)), acquired_handle(acquire(tag2, prov_new(via2, site2), id2))) {{\n\
         \x20   True => {one},\n\
         \x20   False => {zero}\n\
         \x20 }};\n",
        zero = b32(0),
        one = b32(1),
    )
}

fn program(driver: &str) -> String {
    format!("{SUBSTRATE_SRC}\n{}\n{driver}", driver_prelude())
}

/// L1-eval-only assertion (the M-981 convention every prior stage uses — the L0 substitution
/// interpreter is not the point of this differential; the self-hosted `.myc` port is).
fn assert_l1_only_u32(label: &str, src: &str, expected_u32: u32) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));
    let repr_val = l1_core
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: expected a Repr CoreValue, got {l1_core:?}"));
    let got = match repr_val.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("{label}: expected a Bits payload, got {other:?}"),
    };
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the Rust-oracle-computed expected value {expected_u32}"
    );
}

/// A driver call asserting a packed `Binary{32}` verdict of `1` (agreement) — every `sub_*`
/// helper above returns `1` on full agreement, `0` on any mismatch.
fn assert_ok(label: &str, driver_call: &str) {
    let src = program(&format!("fn main() => Binary{{32}} = {driver_call};"));
    assert_l1_only_u32(label, &src, 1);
}

/// The Stage-4 structural gate: `substrate.myc` parses and type-checks green (no driver needed).
#[test]
fn substrate_myc_parses_and_checks() {
    let nodule =
        parse(SUBSTRATE_SRC).unwrap_or_else(|e| panic!("substrate.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("substrate.myc: check failed: {e}"));
}

/// `acquire` + `explain`: the live oracle mints a real handle (reading back its ACTUAL assigned
/// id, per FLAG-substrate-2 — this gate makes no claim about the Rust-side counter's own
/// numbering, only that both sides agree given the SAME concrete id), then the `.myc` port must
/// agree on identity, the next-id thread, freshness, and `explain`'s BYTE-FOR-BYTE output (its
/// format string is pure ASCII, so no FLAG-substrate-4 narrowing applies to this one).
#[test]
fn substrate_myc_matches_oracle_on_acquire_and_explain() {
    let cases: &[(&str, &str, &str)] = &[
        ("disk", "wild:open", "test_site_a"),
        ("net-socket", "graft(cap)", "handler::accept"),
        ("gpu", "wild:alloc", ""),
    ];
    for (tag, via, site) in cases {
        let prov = SubstrateProvenance::new(*via, *site);
        let h = SubstrateHandle::acquire(*tag, prov);
        let id = u32::try_from(h.id()).expect("test-run ids stay well within u32 range");
        let expected_explain = h.explain();

        let label = format!("acquire+explain: tag={tag:?} via={via:?} site={site:?} id={id}");
        let tag_e = escape_myc_string(tag);
        let via_e = escape_myc_string(via);
        let site_e = escape_myc_string(site);
        let explain_e = escape_myc_string(&expected_explain);
        let id_lit = b32(id);

        assert_ok(
            &format!("{label}: id"),
            &format!(r#"sub_acquire_id_ok("{tag_e}", "{via_e}", "{site_e}", {id_lit})"#),
        );
        assert_ok(
            &format!("{label}: next-id thread"),
            &format!(
                r#"sub_acquire_next_ok("{tag_e}", "{via_e}", "{site_e}", {id_lit}, {})"#,
                b32(id + 1)
            ),
        );
        assert_ok(
            &format!("{label}: fresh handle is Live"),
            &format!(r#"sub_acquire_fresh_live("{tag_e}", "{via_e}", "{site_e}", {id_lit})"#),
        );
        assert_ok(
            &format!("{label}: explain == {expected_explain:?}"),
            &format!(
                r#"sub_explain_eq("{tag_e}", "{via_e}", "{site_e}", {id_lit}, "{explain_e}")"#
            ),
        );
    }
}

/// The use-once transition along ONE threaded value: the first `try_consume` succeeds (Live ->
/// Consumed, same identity); the second (on the now-consumed handle) refuses with
/// `AlreadyConsumed{tag, id}` — matching the oracle's own tag/id on that refusal, and matching a
/// Rust-side ASCII-safe MIRROR of `se_display` (FLAG-substrate-4 — NOT the live, non-ASCII
/// `Display` impl).
#[test]
fn substrate_myc_matches_oracle_on_try_consume() {
    fn mirror_se_display(tag: &str, id: u32) -> String {
        format!("double-consume: Substrate{{{tag}}} #{id} was already consumed")
    }

    let cases: &[(&str, &str, &str)] = &[
        ("disk", "wild:open", "test_site_a"),
        ("net-socket", "graft(cap)", "handler::accept"),
    ];
    for (tag, via, site) in cases {
        let prov = SubstrateProvenance::new(*via, *site);
        let h = SubstrateHandle::acquire(*tag, prov);
        let id = u32::try_from(h.id()).expect("test-run ids stay well within u32 range");

        // Oracle sanity: first try_consume Ok + consumed; second Err naming this exact tag/id.
        let h2 = h
            .try_consume()
            .unwrap_or_else(|e| panic!("oracle: first try_consume must be Ok: {e}"));
        assert!(h2.is_consumed(), "oracle: consumed handle reports consumed");
        let err = h2
            .try_consume()
            .expect_err("oracle: second try_consume on a consumed handle must be Err");
        match &err {
            SubstrateError::AlreadyConsumed { tag: etag, id: eid } => {
                assert_eq!(etag, tag, "oracle: AlreadyConsumed names the violated tag");
                assert_eq!(
                    *eid,
                    h.id(),
                    "oracle: AlreadyConsumed names the violated id"
                );
            }
        }

        let label = format!("try_consume: tag={tag:?} via={via:?} site={site:?} id={id}");
        let tag_e = escape_myc_string(tag);
        let via_e = escape_myc_string(via);
        let site_e = escape_myc_string(site);
        let id_lit = b32(id);

        assert_ok(
            &format!("{label}: first try_consume is Ok+consumed"),
            &format!(r#"sub_try_consume_first_ok("{tag_e}", "{via_e}", "{site_e}", {id_lit})"#),
        );
        assert_ok(
            &format!("{label}: second try_consume is Err{{tag,id}}"),
            &format!(r#"sub_try_consume_second_err("{tag_e}", "{via_e}", "{site_e}", {id_lit})"#),
        );
        let want = mirror_se_display(tag, id);
        let want_e = escape_myc_string(&want);
        assert_ok(
            &format!("{label}: se_display == mirror {want:?}"),
            &format!(
                r#"sub_se_display_eq("{tag_e}", "{via_e}", "{site_e}", {id_lit}, "{want_e}")"#
            ),
        );
    }
}

/// `release`: a Live handle releases to `Some(ReleaseEvent{{tag,id,site}})` on the FIRST call
/// (matching a Rust-side ASCII-safe MIRROR of `re_display`, FLAG-substrate-4); an already-consumed
/// handle releases to `None` (never fabricated — G2). FLAG-substrate-6: `release` never returns
/// an updated handle, matching the oracle's own signature exactly.
#[test]
fn substrate_myc_matches_oracle_on_release() {
    fn mirror_re_display(tag: &str, id: u32, site: &str) -> String {
        format!("released: Substrate{{{tag}}} #{id} released at scope exit ({site})")
    }

    let cases: &[(&str, &str, &str, &str)] = &[
        ("disk", "wild:open", "test_site_a", "binding `f`"),
        (
            "net-socket",
            "graft(cap)",
            "handler::accept",
            "binding `conn`",
        ),
    ];
    for (tag, via, site, rsite) in cases {
        let prov = SubstrateProvenance::new(*via, *site);
        let h = SubstrateHandle::acquire(*tag, prov.clone());
        let id = u32::try_from(h.id()).expect("test-run ids stay well within u32 range");

        // Oracle sanity: fresh release is Some with the right fields; a second (post-consume)
        // release on the SAME identity is None.
        let re = h
            .release(*rsite)
            .unwrap_or_else(|| panic!("oracle: release of a Live handle must be Some"));
        assert_eq!(re.tag, *tag);
        assert_eq!(re.id, h.id());
        assert_eq!(re.site, *rsite);

        let h_for_consume = SubstrateHandle::acquire(*tag, prov);
        let id2 = h_for_consume.id();
        let h2 = h_for_consume
            .try_consume()
            .unwrap_or_else(|e| panic!("oracle: try_consume must be Ok: {e}"));
        assert!(
            h2.release(*rsite).is_none(),
            "oracle: release of an already-consumed handle must be None"
        );

        let label = format!("release: tag={tag:?} via={via:?} site={site:?} id={id}");
        let tag_e = escape_myc_string(tag);
        let via_e = escape_myc_string(via);
        let site_e = escape_myc_string(site);
        let rsite_e = escape_myc_string(rsite);
        let id_lit = b32(id);

        assert_ok(
            &format!("{label}: fresh release is Some{{tag,id,site}}"),
            &format!(
                r#"sub_release_fresh_some("{tag_e}", "{via_e}", "{site_e}", {id_lit}, "{rsite_e}")"#
            ),
        );
        let want = mirror_re_display(tag, id, rsite);
        let want_e = escape_myc_string(&want);
        assert_ok(
            &format!("{label}: re_display == mirror {want:?}"),
            &format!(
                r#"sub_re_display_eq("{tag_e}", "{via_e}", "{site_e}", {id_lit}, "{rsite_e}", "{want_e}")"#
            ),
        );

        // The second (post-consume) leg uses id2 (a distinct real identity from `id`).
        let id2_u32 = u32::try_from(id2).expect("test-run ids stay well within u32 range");
        assert_ok(
            &format!("{label}: release after consume is None (id2={id2_u32})"),
            &format!(
                r#"sub_release_after_consume_none("{tag_e}", "{via_e}", "{site_e}", {}, "{rsite_e}")"#,
                b32(id2_u32)
            ),
        );
    }
}

/// `SubstrateHandle`'s identity equality (mirrors `substrate.rs`'s own `PartialEq` doc contract:
/// "two handles are the same resource iff their ids are equal" — `provenance`/`tag` do not enter
/// the comparison once `id` is fixed).
#[test]
fn substrate_myc_matches_oracle_on_handle_identity_equality() {
    let prov = SubstrateProvenance::new("wild:open", "site");
    let h1 = SubstrateHandle::acquire("disk", prov.clone());
    let h2 = SubstrateHandle::acquire("disk", prov);
    // Two acquisitions of "the same" tag are two DISTINCT identities (never equal) — the oracle's
    // own documented contract ("two acquisitions of the same external resource are two distinct
    // handles").
    assert_ne!(
        h1, h2,
        "oracle: two acquisitions are never the same identity"
    );
    let id1 = u32::try_from(h1.id()).expect("test-run ids stay well within u32 range");
    let id2 = u32::try_from(h2.id()).expect("test-run ids stay well within u32 range");
    assert_ne!(id1, id2);

    assert_ok(
        "handle identity: two distinct acquisitions are not sh_eq",
        &format!(
            r#"match sh_eq(acquired_handle(acquire("disk", prov_new("wild:open", "site"), {})), acquired_handle(acquire("disk", prov_new("wild:open", "site"), {}))) {{ False => {}, True => {} }}"#,
            b32(id1),
            b32(id2),
            b32(1),
            b32(0),
        ),
    );
    // A handle is equal to itself (reflexive identity — same id compared to itself).
    assert_ok(
        "handle identity: a handle is sh_eq to itself",
        &format!(
            r#"sub_handle_eq("disk", "wild:open", "site", {}, "disk", "wild:open", "site", {})"#,
            b32(id1),
            b32(id1),
        ),
    );
}
