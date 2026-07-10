//! The **canonical formatter** (M-142; RFC-0001 §4.8; ADR-003).
//!
//! Reformatting is a *projection*: it produces a canonical textual normal form and **never changes
//! a definition's content-addressed identity** (RFC-0001 §4.6; ADR-003). The canonical form
//! α-normalizes binder names (`v0, v1, …`), so two definitions that differ only in names — the
//! essence of a reformatting — render to the *same* text and share the *same* [`content_hash`]. The
//! heavy lifting lives in [`mycelium_core::lower::format`] (the IR-level canonical dump, §4.8); this
//! module is the toolchain-facing entry plus its identity guarantees.
//!
//! [`content_hash`]: mycelium_core::Node::content_hash

use mycelium_core::Node;

/// Format a Core IR node into its canonical textual normal form (α-normalized binders).
#[must_use]
pub fn format(node: &Node) -> String {
    mycelium_core::lower::format(node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{ContentHash, Meta, Payload, Provenance, Repr, Value};

    fn byte() -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    /// `let <binder> = byte in swap(<binder> -> Ternary{6})`, parameterized by the binder name.
    fn def(binder: &str) -> Node {
        Node::Let {
            id: binder.to_owned(),
            bound: Box::new(Node::Const(byte())),
            body: Box::new(Node::Swap {
                src: Box::new(Node::Var(binder.to_owned())),
                target: Repr::Ternary { trits: 6 },
                policy: ContentHash::parse("blake3:round_trip_safe").unwrap(),
            }),
        }
    }

    #[test]
    fn formatting_is_deterministic() {
        assert_eq!(format(&def("a")), format(&def("a")));
    }

    /// ADR-003: reformatting (here, renaming binders) yields the **same canonical text** and leaves
    /// content-addressed identity **unchanged**.
    #[test]
    fn reformatting_preserves_canonical_text_and_identity() {
        let a = def("a");
        let renamed = def("a_very_long_binder_name");
        // The α-normalized canonical text is identical despite the different binder names.
        assert_eq!(format(&a), format(&renamed));
        // And identity is unchanged (names are not hashed; RFC-0001 §4.6 / M-103).
        assert_eq!(a.content_hash(), renamed.content_hash());
    }

    #[test]
    fn formatting_does_not_mutate_identity() {
        let n = def("a");
        let before = n.content_hash();
        let _ = format(&n);
        assert_eq!(before, n.content_hash());
    }

    #[test]
    fn canonical_names_appear_not_source_names() {
        let text = format(&def("my_buffer"));
        assert!(text.contains("let v0 ="), "{text}");
        assert!(text.contains("var v0"), "{text}");
        assert!(!text.contains("my_buffer"), "source name leaked: {text}");
    }

    #[test]
    fn structurally_distinct_defs_format_differently() {
        let swap_def = def("a");
        let const_def = Node::Const(byte());
        assert_ne!(format(&swap_def), format(&const_def));
    }

    #[test]
    fn free_variables_keep_their_names() {
        // A free variable is not α-renamable (it is part of the open term's contract).
        let text = format(&Node::Var("external".to_owned()));
        assert!(text.contains("free external"), "{text}");
    }
}
