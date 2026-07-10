use crate::error::*;
use crate::token::Pos;

#[test]
fn at_alias_equals_new() {
    let p = Pos { line: 1, col: 1 };
    assert_eq!(
        ParseError::at(p, "boom"),
        ParseError::new(p, "boom".to_owned())
    );
    // Accepts an owned String too.
    assert_eq!(
        ParseError::at(p, String::from("x")),
        ParseError::new(p, "x".to_owned())
    );
}
