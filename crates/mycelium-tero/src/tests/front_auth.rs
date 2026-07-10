//! White-box tests for token-scoped auth (M-1017 / DN-87 §6.4): the `token:scope` grammar, the
//! never-silent parse errors, the scope lattice (`refresh ⊇ read`), and the typed authorization
//! outcomes + their mapping to a front error. (`from_env` is a thin wrapper over `parse` reading
//! two fixed env vars; it is exercised at runtime by the binaries, not unit-tested here — setting
//! process env would race the parallel test runner.)

use crate::front::auth::{AuthError, Scope, TokenTable, TokenTableError};
use crate::front::core::FrontError;

#[test]
fn parse_round_trips_a_token_scope_list() {
    let t = TokenTable::parse("reader:read  admin:refresh").unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(
        t.authorize(Some("reader"), Scope::Read).unwrap(),
        Scope::Read
    );
    assert_eq!(
        t.authorize(Some("admin"), Scope::Refresh).unwrap(),
        Scope::Refresh
    );
}

#[test]
fn parse_accepts_comma_and_whitespace_separators() {
    let t = TokenTable::parse("a:read, b:refresh\n c:read").unwrap();
    assert_eq!(t.len(), 3);
}

#[test]
fn parse_rejects_malformed_entries_and_empty_source_never_silently() {
    // No colon, empty token, unknown scope keyword — each an explicit Malformed, never dropped.
    assert!(matches!(
        TokenTable::parse("nocolon"),
        Err(TokenTableError::Malformed(_))
    ));
    assert!(matches!(
        TokenTable::parse(":read"),
        Err(TokenTableError::Malformed(_))
    ));
    assert!(matches!(
        TokenTable::parse("a:superuser"),
        Err(TokenTableError::Malformed(_))
    ));
    // An empty / whitespace-only source is Empty (refuse an accidentally-open server), not Ok.
    assert!(matches!(
        TokenTable::parse("   "),
        Err(TokenTableError::Empty)
    ));
}

#[test]
fn scope_lattice_refresh_is_a_superset_of_read() {
    assert!(Scope::Refresh.allows(Scope::Read));
    assert!(Scope::Refresh.allows(Scope::Refresh));
    assert!(Scope::Read.allows(Scope::Read));
    assert!(!Scope::Read.allows(Scope::Refresh));
}

#[test]
fn authorize_distinguishes_missing_invalid_and_insufficient() {
    let t = TokenTable::parse("reader:read admin:refresh").unwrap();
    // Missing token.
    assert_eq!(t.authorize(None, Scope::Read), Err(AuthError::Missing));
    // Unknown token.
    assert_eq!(
        t.authorize(Some("ghost"), Scope::Read),
        Err(AuthError::Invalid)
    );
    // Valid read token, but the op needs refresh.
    assert_eq!(
        t.authorize(Some("reader"), Scope::Refresh),
        Err(AuthError::InsufficientScope {
            have: Scope::Read,
            need: Scope::Refresh
        })
    );
    // A refresh token satisfies a read op (superset).
    assert_eq!(
        t.authorize(Some("admin"), Scope::Read).unwrap(),
        Scope::Refresh
    );
}

#[test]
fn auth_error_maps_to_the_right_front_error() {
    assert!(matches!(
        FrontError::from(AuthError::Missing),
        FrontError::Unauthorized(_)
    ));
    assert!(matches!(
        FrontError::from(AuthError::Invalid),
        FrontError::Unauthorized(_)
    ));
    assert!(matches!(
        FrontError::from(AuthError::InsufficientScope {
            have: Scope::Read,
            need: Scope::Refresh
        }),
        FrontError::Forbidden(_)
    ));
}
