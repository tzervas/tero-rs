// Tests for `mycelium-cli` lib root (extracted from inline `#[cfg(test)]` per M-797 / CLAUDE.md).
// White-box access via `use crate::*`.
use crate::*;
use std::path::PathBuf;

/// The committed single-nodule fixture for `myc run` v0 (M-908):
/// `tests/fixtures/run-single-nodule/{mycelium-proj.toml,run_hello.myc}`.
fn run_fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/run-single-nodule/mycelium-proj.toml")
}

/// The committed multi-nodule fixture for `myc run` linking (M-909):
/// `tests/fixtures/run-multi-nodule/{mycelium-proj.toml,mathutils.myc,run_multi_nodule.myc}`.
fn run_multi_fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/run-multi-nodule/mycelium-proj.toml")
}

/// The committed H1-capstone fixture (M-914; E28-1 `enb` — signed/unsigned integer ops, the
/// string literal + `hash_blake3`/`bytes_eq`, and the `@forage` D-lite placement-policy surface):
/// `tests/fixtures/run-h1-capstone/{mycelium-proj.toml,h1_signed.myc,h1_unsigned.myc,
/// h1_strings.myc,run_h1_capstone.myc}`.
fn run_h1_capstone_fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/run-h1-capstone/mycelium-proj.toml")
}

/// The committed H1-capstone float-leg fixture (M-914; E28-1 `enb` — the `Float` value form +
/// `flt.*` ops, kept separate from `run-h1-capstone` because `bit.and` has no defined
/// ε-propagation rule for a non-`Exact` operand and every `flt.*` result is `Empirical` — see
/// `run_h1_float.myc`'s doc comment): `tests/fixtures/run-h1-float/{mycelium-proj.toml,
/// run_h1_float.myc}`.
fn run_h1_float_fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run-h1-float/mycelium-proj.toml")
}

/// Write a minimal `mycelium-proj.toml` (matching the ad-hoc manifests already used by the
/// `run_reports_a_located_parse_error_not_a_panic` / `run_refuses_zero_myc_sources_explicitly`
/// tests below) into `dir`, named `name`.
fn write_manifest(dir: &std::path::Path, name: &str) {
    std::fs::write(
        dir.join("mycelium-proj.toml"),
        format!(
            "[project]\nname=\"{name}\"\nkind=\"phylum\"\nversion=\"0.1.0\"\nlicense=\"MIT\"\nsummary=\"s\"\n\n[surface]\nexports=[\"{name}\"]\n"
        ),
    )
    .unwrap();
}

fn scratch(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "myc-cli-{tag}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn init_scaffolds_a_buildable_checkable_phylum() {
    let parent = scratch("init");
    let files = init(&parent, "acme").expect("init succeeds");
    assert_eq!(files.len(), 2);
    let manifest = parent.join("acme").join("mycelium-proj.toml");
    assert!(manifest.exists());

    // The scaffold must BUILD (spore) ...
    let (spore, _descriptor) = build(&manifest).expect("scaffold builds");
    assert_eq!(spore.name, "acme");
    assert!(spore.id.as_str().starts_with("blake3:"));

    // ... and type-CHECK cleanly (parse + check).
    let report = check_project(&manifest).expect("walk succeeds");
    assert!(
        report.ok(),
        "scaffold should check clean: {:?}",
        report.failures
    );
    assert_eq!(report.checked.len(), 1);
}

#[test]
fn init_refuses_a_bad_name_without_normalizing() {
    let parent = scratch("badname");
    for bad in ["Acme", "1geo", "geo-metry", "", "geo.core"] {
        let err = init(&parent, bad).unwrap_err();
        assert_eq!(err.code, "myc-init-name", "{bad:?} should be rejected");
        assert_eq!(err.exit, 64);
    }
}

#[test]
fn init_never_overwrites_an_existing_project() {
    let parent = scratch("noclobber");
    init(&parent, "acme").unwrap();
    let err = init(&parent, "acme").unwrap_err();
    assert_eq!(err.code, "myc-init-exists");
}

#[test]
fn check_reports_a_located_parse_error_not_a_panic() {
    let parent = scratch("badsrc");
    init(&parent, "acme").unwrap();
    let dir = parent.join("acme");
    // Introduce a syntax error in a second nodule.
    std::fs::write(dir.join("broken.myc"), "nodule broken\nfn f() = §\n").unwrap();
    let report = check_project(&dir.join("mycelium-proj.toml")).expect("walk ok");
    assert!(!report.ok());
    let parse_fail = report
        .failures
        .iter()
        .find(|r| r.code == "myc-parse")
        .expect("a parse failure is reported");
    // DN-22: the failure carries a location and a help line, never an opaque panic.
    assert!(parse_fail.location.as_ref().unwrap().contains("broken.myc"));
    assert!(parse_fail.help.is_some());
}

#[test]
fn run_executes_a_committed_single_nodule_fixture_end_to_end() {
    // M-908: `myc run` on the committed single-nodule fixture actually runs `main` through the
    // reference interpreter (`not(0b1010_1010)` == `0b0101_0101`), never a stub / silent no-op.
    let report = run(&run_fixture_manifest()).expect("the fixture runs end-to-end");
    assert_eq!(report.entry, "main");
    assert_eq!(report.source, "run_hello.myc");
    // v0 rendering is a `Debug` dump of the interpreter's `Value` — assert on its substance (the
    // bitwise-negated payload), not a brittle exact string match on internal `Debug` formatting.
    assert!(
        report
            .rendered
            .contains("false, true, false, true, false, true, false, true"),
        "rendered result should show 0b0101_0101's bits: {}",
        report.rendered
    );
}

#[test]
fn run_refuses_zero_myc_sources_explicitly() {
    let parent = scratch("run-nosource");
    std::fs::write(
        parent.join("mycelium-proj.toml"),
        "[project]\nname=\"empty\"\nkind=\"phylum\"\nversion=\"0.1.0\"\nlicense=\"MIT\"\nsummary=\"s\"\n\n[surface]\nexports=[\"empty\"]\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-no-source");
    assert!(err.help.is_some());
}

#[test]
fn run_multi_nodule_projects_now_link_and_run_not_a_blanket_refusal() {
    // M-909 landed: what M-908 v0 refused wholesale as `myc-run-multi-nodule` now actually links +
    // runs (or fails with a specific, named diagnostic) — never a blanket "not yet wired" refusal
    // (G2). Neither of these two unrelated nodules declares `main`, so the specific diagnostic is
    // the M-909 `myc-run-no-entry` (the same code the single-nodule path already used).
    let parent = scratch("run-multinodule");
    init(&parent, "acme").unwrap();
    let dir = parent.join("acme");
    std::fs::write(
        dir.join("second.myc"),
        "// nodule: second\nnodule second;\n\nfn other() => Binary{8} = 0b0000_0000;\n",
    )
    .unwrap();
    let err = run(&dir.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-no-entry");
    assert!(err.help.is_some());
}

#[test]
fn run_executes_a_committed_multi_nodule_fixture_end_to_end() {
    // M-909: the entry nodule's `main` calls an imported `pub` fn (`mathutils.quadruple`) whose
    // body calls a *private* helper of `mathutils` the entry never imports — a real transitive
    // cross-nodule link (the v0 flatten-by-name `Env` merge exists exactly for this), not just a
    // same-file multi-fn call. `quadruple(0b0000_0011)` == 12 == `0b0000_1100`.
    let report = run(&run_multi_fixture_manifest())
        .expect("the multi-nodule fixture links and runs end-to-end");
    assert_eq!(report.entry, "main");
    assert_eq!(report.source, "run_multi_nodule.myc");
    assert!(
        report
            .rendered
            .contains("false, false, false, false, true, true, false, false"),
        "rendered result should show 0b0000_1100's bits: {}",
        report.rendered
    );
}

#[test]
fn run_executes_the_h1_capstone_fixture_end_to_end() {
    // M-914 (E28-1 `enb` H1 capstone): the entry nodule's `main` folds the signed integer op set
    // (`h1_signed`), the unsigned integer op set (`h1_unsigned`), and the string-literal +
    // `hash_blake3`/`bytes_eq` check (`h1_strings`, riding inside a `colony { @forage(…) hypha …
    // }`) into one `Binary{1}` via `and` — every sub-check is a hand-verified worked example (see
    // `crates/mycelium-l1/tests/enablement.rs`), so this must render `true`; a single wrong
    // constant anywhere would flip it to `false`, never silently pass (G2/VR-5).
    let report = run(&run_h1_capstone_fixture_manifest())
        .expect("the H1 capstone fixture links and runs end-to-end");
    assert_eq!(report.entry, "main");
    assert_eq!(report.source, "run_h1_capstone.myc");
    assert!(
        report.rendered.contains("Bits([true])"),
        "the H1 capstone's folded and-chain must render true (every sub-check passes): {}",
        report.rendered
    );
}

#[test]
fn run_executes_the_h1_float_fixture_end_to_end() {
    // M-914: the `Float` value form + `flt.*` op set — kept as its OWN fixture (not folded into
    // `run-h1-capstone`'s `and`-chain) because `bit.and` has no defined ε-propagation rule for a
    // non-`Exact` operand (ADR-010/M-204) and every `flt.*` result is `Empirical` (ADR-040 §2.6) —
    // see `run_h1_float.myc`'s doc comment for the discovered constraint. `main` is a single
    // `flt_eq(flt_add(flt_mul(1.5, 2.0), 0.25), 3.25)`, so this must render `true`.
    let report =
        run(&run_h1_float_fixture_manifest()).expect("the H1 float-leg fixture runs end-to-end");
    assert_eq!(report.entry, "main");
    assert_eq!(report.source, "run_h1_float.myc");
    assert!(
        report.rendered.contains("Bits([true])"),
        "the float composition (1.5*2.0)+0.25 == 3.25 must render true: {}",
        report.rendered
    );
    assert!(
        report.rendered.contains("guarantee: Empirical"),
        "the float leg must stay honestly Empirical-tagged (ADR-040 §2.6), never upgraded: {}",
        report.rendered
    );
}

#[test]
fn run_refuses_duplicate_nodule_paths_explicitly() {
    // M-909: two files both declaring `nodule shared;` would silently collide in check_phylum's
    // qualified export table if unguarded — refused before check_phylum even runs (G2).
    let parent = scratch("run-nodule-dup");
    write_manifest(&parent, "dup");
    std::fs::write(
        parent.join("a.myc"),
        "nodule shared;\nfn f() => Binary{8} = 0b0000_0000;\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("b.myc"),
        "nodule shared;\nfn main() => Binary{8} = 0b0000_0001;\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-nodule-duplicate");
    assert!(err.message.contains("shared"));
    assert!(err.help.is_some());
}

#[test]
fn run_refuses_an_unresolved_nodule_use_explicitly() {
    // M-909: `use ghost.helper;` names a nodule with no corresponding `.myc` file in the project —
    // an explicit, named refusal rather than an opaque "unknown name" from check_phylum.
    let parent = scratch("run-nodule-unresolved");
    write_manifest(&parent, "unresolved");
    std::fs::write(
        parent.join("main.myc"),
        "nodule entry;\nuse ghost.helper;\nfn main() => Binary{8} = helper(0b0000_0001);\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("other.myc"),
        "nodule other;\nfn noop() => Binary{8} = 0b0000_0000;\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-nodule-unresolved");
    assert!(err.message.contains("ghost"));
}

#[test]
fn run_refuses_a_cyclic_nodule_use_graph_explicitly() {
    // M-909, `Declared` v0 policy: nodule `a` `use`s `b`, and `b` `use`s `a` — myc run v0 requires
    // an acyclic nodule-level `use` graph (see `run_multi_nodule`'s doc for why this is a v0 CLI
    // choice, not a `check_phylum` limitation).
    let parent = scratch("run-nodule-cycle");
    write_manifest(&parent, "cyclic");
    std::fs::write(
        parent.join("a.myc"),
        "nodule a;\nuse b.g;\npub fn f(x: Binary{8}) => Binary{8} = x;\nfn main() => Binary{8} = g(0b0000_0001);\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("b.myc"),
        "nodule b;\nuse a.f;\npub fn g(x: Binary{8}) => Binary{8} = f(x);\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-nodule-cyclic");
    assert!(err.help.is_some());
}

#[test]
fn run_refuses_an_ambiguous_multi_nodule_entry_never_guessing() {
    // M-909: two different, unrelated nodules each declare a nullary `main` — `myc run` must never
    // guess which one is "the" entry (G2/VR-5).
    let parent = scratch("run-nodule-ambiguous-entry");
    write_manifest(&parent, "ambiguous");
    std::fs::write(
        parent.join("a.myc"),
        "nodule a;\nfn main() => Binary{8} = 0b0000_0000;\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("b.myc"),
        "nodule b;\nfn main() => Binary{8} = 0b0000_0001;\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-entry-ambiguous");
    assert!(err.message.contains('a') && err.message.contains('b'));
}

#[test]
fn run_refuses_a_cross_nodule_fn_name_collision_when_merging_for_elaboration() {
    // Safety net for the v0 flatten-by-name link (`run_multi_nodule`'s doc, step 5): two different
    // nodules privately declaring the same simple name with different bodies cannot be
    // unambiguously merged into one `Env` for elaboration — refused explicitly rather than silently
    // picking a winner (G2), even though neither is reachable from `main`.
    let parent = scratch("run-nodule-fn-collision");
    write_manifest(&parent, "collide");
    std::fs::write(
        parent.join("runner.myc"),
        "nodule runner;\nfn main() => Binary{8} = 0b0000_0000;\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("x.myc"),
        "nodule x;\nfn helper(v: Binary{8}) => Binary{8} = v;\n",
    )
    .unwrap();
    std::fs::write(
        parent.join("y.myc"),
        "nodule y;\nfn helper(v: Binary{8}) => Binary{8} = not(v);\n",
    )
    .unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-nodule-fn-collision");
    assert!(err.message.contains("helper"));
}

#[test]
fn run_refuses_a_missing_main_entry_never_guessing_another_fn() {
    // `myc init`'s scaffold defines `answer()`, not `main()` — v0 `myc run` must refuse rather
    // than silently running whatever function it finds (G2/VR-5).
    let parent = scratch("run-noentry");
    init(&parent, "acme").unwrap();
    let err = run(&parent.join("acme").join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-run-no-entry");
    assert!(err.help.unwrap().contains("answer"));
}

#[test]
fn run_reports_a_located_parse_error_not_a_panic() {
    let parent = scratch("run-badsrc");
    std::fs::write(
        parent.join("mycelium-proj.toml"),
        "[project]\nname=\"badsrc\"\nkind=\"phylum\"\nversion=\"0.1.0\"\nlicense=\"MIT\"\nsummary=\"s\"\n\n[surface]\nexports=[\"badsrc\"]\n",
    )
    .unwrap();
    std::fs::write(parent.join("badsrc.myc"), "nodule badsrc\nfn f() = §\n").unwrap();
    let err = run(&parent.join("mycelium-proj.toml")).unwrap_err();
    assert_eq!(err.code, "myc-parse");
    assert!(err.location.as_ref().unwrap().contains("badsrc.myc"));
}

#[test]
fn report_renders_the_dn22_structured_form() {
    let r = Report::new("myc-parse", "unexpected token", 65)
        .at("a.myc:3:7")
        .help("remove the stray character");
    let s = r.render();
    assert!(s.starts_with("error[myc-parse]: unexpected token"));
    assert!(s.contains("--> a.myc:3:7"));
    assert!(s.contains("help: remove the stray character"));
}
