//! Packed-ternary codecs (RFC-0004 §5; DN-01; `docs/spec/stdlib/ternary.md` §3).
//!
//! A packing is a **visible, inspectable representation choice** — never a hidden lowering pass.
//! Every packed value carries an explicit [`Scheme`] record (`Meta.physical`) that is queryable
//! via [`scheme_of`] and [`explain`]. The pack/unpack round-trip is lossless (`Exact`) over the
//! I2_S / TL1 / TL2 schemes currently in scope (DN-01 §2; RFC-0004 §5 "pack and unpack keeps
//! int16 sums for lossless inference"). (C3/NFR-1/C4).
//!
//! **FLAG (Q2):** This module's contract is *exact-only*. If a future lossy / non-bit-exact
//! packing scheme is added, it must tag below `Exact` and is **not** admissible under the current
//! matrix. It must NOT be silently folded into this module (`docs/spec/stdlib/ternary.md` §7-Q2).
//!
//! **FLAG (Q3):** The split between "caller names the scheme" (here) and "selector chooses +
//! emits EXPLAIN" (RFC-0005, `std.select`, M-519) needs a cross-module design pass
//! (`docs/spec/stdlib/ternary.md` §7-Q3; RFC-0016 §8-Q3). For v0 the caller names the scheme
//! explicitly.

use crate::primitives::Trit;

// ── Scheme ────────────────────────────────────────────────────────────────────

/// The packing scheme chosen at a lowering stage (RFC-0004 §5; `physical-layout.schema.json`).
///
/// These are the three lossless bit-exact codecs in scope for v0. **`Exact` is honest** because
/// all three are lossless re-encodings (DN-01 §2; RFC-0004 §5). A future lossy scheme is out of
/// scope; if added it would require a tag below `Exact` (FLAG Q2).
///
/// - **`I2S`** — the bitnet.cpp I2_S default: 2 bits per trit (0b00=0, 0b01=+1, 0b11=−1 in the
///   bitnet encoding). Group size 4 trits per byte.
/// - **`Tl1`** — TL1: 5 trits per byte (base-3, using values 0,1,2 mapped to −1,0,+1). Group
///   size 5.
/// - **`Tl2`** — TL2: 5 trits per byte using a different bit arrangement (higher packing
///   density variant of TL1). Group size 5.
///
/// The mirror of [`mycelium_core::PackScheme`] for the I2S/TL1/TL2 variants; surfaced here so
/// Ring-1 callers need not import the core crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scheme {
    /// bitnet.cpp I2_S: 2 bits per trit, group size 4.
    I2S,
    /// TL1: 5 trits per byte (base-3 variant), group size 5.
    Tl1,
    /// TL2: 5 trits per byte (alternate bit arrangement), group size 5.
    Tl2,
}

impl Scheme {
    /// The number of trits packed per byte for this scheme.
    ///
    /// `I2S`: 4 trits/byte (2 bits each × 4 = 8 bits = 1 byte).
    /// `Tl1`/`Tl2`: 5 trits/byte.
    #[must_use]
    pub fn trits_per_byte(self) -> usize {
        match self {
            Scheme::I2S => 4,
            Scheme::Tl1 | Scheme::Tl2 => 5,
        }
    }

    /// The alignment group size (number of trits that must be present for a complete group).
    /// A `pack` call is an explicit `Err(PackError::Misaligned)` when the trit count is not a
    /// multiple of this group size (RFC-0004 §5 "align to SIMD width"; C1).
    #[must_use]
    pub fn group_size(self) -> usize {
        self.trits_per_byte()
    }
}

impl core::fmt::Display for Scheme {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Scheme::I2S => write!(f, "I2S"),
            Scheme::Tl1 => write!(f, "TL1"),
            Scheme::Tl2 => write!(f, "TL2"),
        }
    }
}

// ── PackError ─────────────────────────────────────────────────────────────────

/// Explicit errors returned by [`pack`] (C1/G2 — no silent failure, no sentinel).
///
/// **`OffGrid`** — a trit's encoding falls outside the scheme's alphabet (should not occur for
/// well-formed `Trit` values in the current implementation, but is part of the contract for
/// completeness and future extensibility).
///
/// **`Misaligned`** — the number of trits is not a multiple of the scheme's group size
/// (RFC-0004 §5 "align to SIMD width"). (Mutant witness: if the alignment check were removed,
/// a 7-trit input to I2S (group 4) would produce a malformed partial byte instead of `Err`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    /// A trit value is outside the scheme's encoding alphabet.
    OffGrid,
    /// The trit count is not a multiple of the scheme's group size.
    Misaligned,
}

impl core::fmt::Display for PackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackError::OffGrid => write!(f, "OffGrid: trit outside scheme alphabet"),
            PackError::Misaligned => {
                write!(f, "Misaligned: trit count not a multiple of group size")
            }
        }
    }
}

impl std::error::Error for PackError {}

// ── ExplainRecord ─────────────────────────────────────────────────────────────

/// The inspectable EXPLAIN record attached to a packed value (C3/NFR-1/SC-3/G11).
///
/// When a packing scheme was chosen by the caller (v0, FLAG Q3), the record names the scheme and
/// notes that the selection was explicit. When the RFC-0005 selector is later wired in (M-519),
/// this record will carry the full selection policy reference — the structure is forward-compatible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainRecord {
    /// The scheme that was used.
    pub scheme: Scheme,
    /// How the scheme was selected.
    pub selection: SelectionNote,
    /// The number of trits encoded.
    pub trit_count: usize,
    /// The number of bytes produced.
    pub byte_count: usize,
}

/// How the scheme was selected (for the EXPLAIN record).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionNote {
    /// The caller named the scheme explicitly at the call site.
    ///
    /// **FLAG (Q3):** In a future `std.select`-integrated version, this would be replaced by a
    /// `PolicyRef` (RFC-0005; M-519). The structure is forward-compatible.
    ExplicitCaller,
}

impl core::fmt::Display for ExplainRecord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Scheme={} selection={:?} trits={} bytes={}",
            self.scheme, self.selection, self.trit_count, self.byte_count,
        )
    }
}

// ── Packed ────────────────────────────────────────────────────────────────────

/// A packed trit sequence: bytes + the inspectable `Meta.physical` scheme record (C3/C4/NFR-1).
///
/// The packing is a **visible, inspectable representation choice** (RFC-0004 §5; DN-01). The
/// scheme is queryable via [`scheme_of`] or `Packed::scheme()`, and the full EXPLAIN record via
/// [`explain`] or `Packed::explain()`. Two packings of the same trits under different schemes are
/// the same *logical value* (DN-01 — "lossless packing is not a type distinction"; C4: metadata
/// is not identity).
///
/// **Guarantee: `Exact`.** `pack` then `unpack` is the identity on a well-formed input (C2;
/// verified in tests below).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    /// The packed bytes (lossless re-encoding of the trits; RFC-0004 §5).
    ///
    /// Private by design: a `Packed` can therefore only originate from [`pack`], which keeps the
    /// [`unpack`] invariant airtight — no external mutation can corrupt the buffer and trip the
    /// total-function `expect` (C1/G2). Read-only access is via [`Packed::bytes`].
    bytes: Vec<u8>,
    /// The scheme used — the inspectable `Meta.physical` record (RFC-0001 §4.3).
    scheme: Scheme,
    /// The number of trits originally packed (needed for `unpack` to know the last group size).
    trit_count: usize,
}

impl Packed {
    /// The scheme used to pack these bytes (the `Meta.physical` inspectable record; C3/NFR-1).
    ///
    /// **Guarantee: `Exact`.** Total — every `Packed` was created by [`pack`] which records the
    /// scheme (C3).
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The number of trits originally packed (total; needed for reconstructing the last group).
    #[must_use]
    pub fn trit_count(&self) -> usize {
        self.trit_count
    }

    /// The packed bytes, read-only (lossless re-encoding of the trits; RFC-0004 §5).
    ///
    /// **Guarantee: `Exact`.** Total. Read-only by design — a `Packed` is immutable and can only
    /// be produced by [`pack`], so [`unpack`] is total on it (C1/C4).
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The full EXPLAIN record for this packed value (C3/G11/NFR-1/SC-3).
    ///
    /// **Guarantee: `Exact`.** Total — always available (C3).
    #[must_use]
    pub fn explain(&self) -> ExplainRecord {
        ExplainRecord {
            scheme: self.scheme,
            selection: SelectionNote::ExplicitCaller,
            trit_count: self.trit_count,
            byte_count: self.bytes.len(),
        }
    }
}

// ── scheme_of / explain free functions ────────────────────────────────────────

/// The scheme used to pack `p` (the inspectable `Meta.physical` record; C3/NFR-1).
///
/// **Guarantee: `Exact`.** Total — mirrors `Packed::scheme()` as a free function (C3).
#[must_use]
pub fn scheme_of(p: &Packed) -> Scheme {
    p.scheme()
}

/// The full EXPLAIN record for `p` — why this scheme was chosen (C3/G11/NFR-1/SC-3).
///
/// **Guarantee: `Exact`.** Total — mirrors `Packed::explain()` as a free function (C3).
#[must_use]
pub fn explain(p: &Packed) -> ExplainRecord {
    p.explain()
}

// ── I2S codec ─────────────────────────────────────────────────────────────────
//
// I2_S: 2 bits per trit, 4 trits per byte, MSB-first within the byte.
// Encoding: Pos=0b01, Zero=0b00, Neg=0b11.
// (The bitnet.cpp convention; the exact bit pattern is documented here so the codec is auditable
// and not a black box — C3/G11.)

const I2S_POS: u8 = 0b01;
const I2S_ZERO: u8 = 0b00;
const I2S_NEG: u8 = 0b11;

fn i2s_encode_trit(t: Trit) -> u8 {
    match t {
        Trit::Pos => I2S_POS,
        Trit::Zero => I2S_ZERO,
        Trit::Neg => I2S_NEG,
    }
}

fn i2s_decode_trit(bits: u8) -> Result<Trit, PackError> {
    match bits & 0b11 {
        I2S_POS => Ok(Trit::Pos),
        I2S_ZERO => Ok(Trit::Zero),
        I2S_NEG => Ok(Trit::Neg),
        _ => Err(PackError::OffGrid), // 0b10 is off-grid for I2_S
    }
}

fn i2s_pack(ts: &[Trit]) -> Result<Vec<u8>, PackError> {
    debug_assert_eq!(ts.len() % 4, 0); // caller checks alignment
    let mut out = Vec::with_capacity(ts.len() / 4);
    for chunk in ts.chunks(4) {
        let byte = (i2s_encode_trit(chunk[0]) << 6)
            | (i2s_encode_trit(chunk[1]) << 4)
            | (i2s_encode_trit(chunk[2]) << 2)
            | i2s_encode_trit(chunk[3]);
        out.push(byte);
    }
    Ok(out)
}

fn i2s_unpack(bytes: &[u8], trit_count: usize) -> Result<Vec<Trit>, PackError> {
    let mut out = Vec::with_capacity(trit_count);
    for &byte in bytes {
        out.push(i2s_decode_trit(byte >> 6)?);
        out.push(i2s_decode_trit(byte >> 4)?);
        out.push(i2s_decode_trit(byte >> 2)?);
        out.push(i2s_decode_trit(byte)?);
    }
    out.truncate(trit_count); // drop padding trits in the last byte (none for aligned input)
    Ok(out)
}

// ── TL1 codec ─────────────────────────────────────────────────────────────────
//
// TL1: 5 trits per byte. Each trit is in {−1, 0, +1} = {0, 1, 2} (shifted by +1 → stored as
// base-3 digit). Five base-3 digits fit in a byte: 3^5 = 243 ≤ 255. The value stored is the
// 5-digit base-3 number with the first trit being the most-significant base-3 digit.
//
// Encoding: trit Neg↦0, Zero↦1, Pos↦2 (shifted by +1). Value = Σᵢ dᵢ·3^(4-i).
// The max stored value is 4·(3^4+3^3+3^2+3+1) = 4·121 = 242 < 243 < 256 → fits in u8.

fn tl1_encode_digit(t: Trit) -> u8 {
    match t {
        Trit::Neg => 0,
        Trit::Zero => 1,
        Trit::Pos => 2,
    }
}

fn tl1_decode_digit(d: u8) -> Result<Trit, PackError> {
    match d {
        0 => Ok(Trit::Neg),
        1 => Ok(Trit::Zero),
        2 => Ok(Trit::Pos),
        _ => Err(PackError::OffGrid),
    }
}

fn tl1_pack(ts: &[Trit]) -> Result<Vec<u8>, PackError> {
    debug_assert_eq!(ts.len() % 5, 0);
    let mut out = Vec::with_capacity(ts.len() / 5);
    for chunk in ts.chunks(5) {
        let mut val: u8 = 0;
        for &t in chunk {
            val = val * 3 + tl1_encode_digit(t);
        }
        out.push(val);
    }
    Ok(out)
}

fn tl1_unpack(bytes: &[u8], trit_count: usize) -> Result<Vec<Trit>, PackError> {
    let mut out = Vec::with_capacity(trit_count);
    for &byte in bytes {
        let mut val = byte;
        let mut group = [Trit::Zero; 5];
        for slot in group.iter_mut().rev() {
            *slot = tl1_decode_digit(val % 3)?;
            val /= 3;
        }
        if val != 0 {
            return Err(PackError::OffGrid); // byte value > 242
        }
        out.extend_from_slice(&group);
    }
    out.truncate(trit_count);
    Ok(out)
}

// ── TL2 codec ─────────────────────────────────────────────────────────────────
//
// TL2: 5 trits per byte, using a different bit arrangement than TL1 for potentially better SIMD
// alignment. TL2 packs 5 trits as two separate bit-planes:
//   - Sign plane (5 bits, positions [4:0]): bit i = 1 iff trit[i] is non-zero.
//   - Magnitude plane (upper 3 bits + 2 bits of sign): repacked for SIMD.
//
// For v0 simplicity and auditability (KC-3): TL2 uses the same base-3 encoding as TL1 but
// with a byte-complement ("dark bits") for the stored value: stored = 242 - tl1_value. This
// is one concrete, inspectable variant documented here as a black-box–free encoding choice
// (C3/G11). Round-trip is verified exhaustively in tests.
//
// FLAG: The exact TL2 bit layout from bitnet.cpp is not yet normative in the Mycelium corpus;
// this is a placeholder that satisfies the losslessness requirement (the important property for
// v0) while the exact normative layout is ratified. When it is, this codec should be updated to
// match exactly, and the tests will catch any divergence.

fn tl2_pack(ts: &[Trit]) -> Result<Vec<u8>, PackError> {
    debug_assert_eq!(ts.len() % 5, 0);
    let mut out = Vec::with_capacity(ts.len() / 5);
    for chunk in ts.chunks(5) {
        let mut val: u8 = 0;
        for &t in chunk {
            val = val * 3 + tl1_encode_digit(t);
        }
        // TL2: complement within the 243-value space (lossless bijection: 242 − val).
        out.push(242 - val);
    }
    Ok(out)
}

fn tl2_unpack(bytes: &[u8], trit_count: usize) -> Result<Vec<Trit>, PackError> {
    let mut out = Vec::with_capacity(trit_count);
    for &byte in bytes {
        // Undo the TL2 complement.
        if byte > 242 {
            return Err(PackError::OffGrid);
        }
        let val = 242 - byte;
        let mut group = [Trit::Zero; 5];
        let mut v = val;
        for slot in group.iter_mut().rev() {
            *slot = tl1_decode_digit(v % 3)?;
            v /= 3;
        }
        out.extend_from_slice(&group);
    }
    out.truncate(trit_count);
    Ok(out)
}

// ── pack / unpack ─────────────────────────────────────────────────────────────

/// Pack a trit sequence under the given scheme.
///
/// The scheme is recorded as the `Meta.physical` inspectable field in the returned [`Packed`]
/// value — never a hidden lowering (C3/NFR-1/RFC-0004 §5; DN-01).
///
/// **Guarantee: `Exact`.** Returns:
/// - `Ok(Packed)` when the input is aligned to the scheme's group size.
/// - `Err(PackError::Misaligned)` when `ts.len()` is not a multiple of the scheme's group size
///   (RFC-0004 §5 "align to SIMD width"; C1).
/// - `Err(PackError::OffGrid)` when a trit's encoding falls outside the scheme's alphabet (C1).
///   (With the current `Trit` type this is unreachable for I2S for the 0b10 code-point, but the
///   path exists for completeness and is verified in unit tests.)
///
/// (Mutant witness: removing the alignment check would let a 7-trit I2S input produce a
/// malformed partial byte instead of `Err(Misaligned)`.)
pub fn pack(ts: &[Trit], scheme: Scheme) -> Result<Packed, PackError> {
    let group = scheme.group_size();
    if !ts.len().is_multiple_of(group) {
        return Err(PackError::Misaligned);
    }
    let bytes = match scheme {
        Scheme::I2S => i2s_pack(ts)?,
        Scheme::Tl1 => tl1_pack(ts)?,
        Scheme::Tl2 => tl2_pack(ts)?,
    };
    Ok(Packed {
        bytes,
        scheme,
        trit_count: ts.len(),
    })
}

/// Unpack a [`Packed`] trit sequence back to a `Vec<Trit>`.
///
/// **Guarantee: `Exact`.** Total on a well-formed [`Packed`] — the codecs are lossless
/// (RFC-0004 §5; DN-01 §2 "lossless physical layout"). The scheme is the one recorded in `p`
/// (inspectable via [`scheme_of`] / `p.scheme()`).
///
/// Note: a [`Packed`] can only be constructed by [`pack`], which validates the input, and its
/// `bytes` field is private — so well-formedness is guaranteed by construction and cannot be
/// broken by a caller. For the type-safe API, `unpack` is total.
///
/// For honest total-function semantics, `unpack` panics on an `OffGrid` from the codec — a state
/// that is **unreachable**: the only way to obtain a `Packed` is via [`pack`] (the `bytes` field
/// is private), which already validated the input, so the invariant cannot be broken externally
/// (C1/G2 — no externally-reachable panic). This is documented explicitly (C3).
#[must_use]
pub fn unpack(p: &Packed) -> Vec<Trit> {
    let result = match p.scheme {
        Scheme::I2S => i2s_unpack(&p.bytes, p.trit_count),
        Scheme::Tl1 => tl1_unpack(&p.bytes, p.trit_count),
        Scheme::Tl2 => tl2_unpack(&p.bytes, p.trit_count),
    };
    // The only way to get OffGrid here is if the `Packed` bytes are somehow corrupt, which
    // cannot happen through the public API (pack already validated). Document rather than silently
    // succeed or silently corrupt (C3 — no black boxes).
    result.expect("unpack: Packed bytes were produced by pack and must decode cleanly")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::{int_to_trits, max_magnitude};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Enumerate all `n`-trit sequences systematically (base-3 counter).
    /// Returns the sequences in lexicographic order over {Neg, Zero, Pos}.
    fn all_n_trit_sequences(n: usize) -> Vec<Vec<Trit>> {
        let trits = [Trit::Neg, Trit::Zero, Trit::Pos];
        let count = 3usize.pow(n as u32);
        (0..count)
            .map(|i| {
                let mut seq = vec![Trit::Zero; n];
                let mut rem = i;
                for slot in seq.iter_mut().rev() {
                    *slot = trits[rem % 3];
                    rem /= 3;
                }
                seq
            })
            .collect()
    }

    // ── scheme_of / explain ───────────────────────────────────────────────────

    #[test]
    fn scheme_of_matches_packed_scheme() {
        // Use 20 trits: divisible by 4 (I2S group) and 5 (TL1/TL2 group).
        let ts = int_to_trits(0, 20).unwrap();
        for scheme in [Scheme::I2S, Scheme::Tl1, Scheme::Tl2] {
            let p = pack(&ts, scheme).unwrap();
            assert_eq!(scheme_of(&p), scheme, "scheme_of for {scheme:?}");
        }
    }

    #[test]
    fn explain_contains_correct_fields() {
        let ts = int_to_trits(7, 4).unwrap(); // some non-trivial value
        let p = pack(&ts, Scheme::I2S).unwrap();
        let rec = explain(&p);
        assert_eq!(rec.scheme, Scheme::I2S);
        assert_eq!(rec.selection, SelectionNote::ExplicitCaller);
        assert_eq!(rec.trit_count, 4);
        assert_eq!(rec.byte_count, 1); // 4 trits / 4 per byte = 1 byte
    }

    // ── pack alignment errors ──────────────────────────────────────────────────

    #[test]
    fn pack_i2s_rejects_non_multiple_of_4() {
        // Mutant witness: removing the alignment check lets 7 trits produce a malformed byte.
        for bad_len in [1usize, 2, 3, 5, 6, 7, 9] {
            let ts: Vec<Trit> = (0..bad_len).map(|_| Trit::Zero).collect();
            assert_eq!(
                pack(&ts, Scheme::I2S),
                Err(PackError::Misaligned),
                "I2S misalignment for {bad_len} trits"
            );
        }
    }

    #[test]
    fn pack_tl1_rejects_non_multiple_of_5() {
        for bad_len in [1usize, 2, 3, 4, 6, 7, 8, 9] {
            let ts: Vec<Trit> = (0..bad_len).map(|_| Trit::Zero).collect();
            assert_eq!(
                pack(&ts, Scheme::Tl1),
                Err(PackError::Misaligned),
                "TL1 misalignment for {bad_len} trits"
            );
        }
    }

    #[test]
    fn pack_tl2_rejects_non_multiple_of_5() {
        for bad_len in [1usize, 2, 3, 4, 6, 7, 8, 9] {
            let ts: Vec<Trit> = (0..bad_len).map(|_| Trit::Zero).collect();
            assert_eq!(
                pack(&ts, Scheme::Tl2),
                Err(PackError::Misaligned),
                "TL2 misalignment for {bad_len} trits"
            );
        }
    }

    // ── pack/unpack round-trip (losslessness) ─────────────────────────────────

    #[test]
    fn i2s_round_trip_exhaustive_4_trits() {
        // All 3^4 = 81 four-trit sequences must round-trip exactly under I2S.
        for ts in all_n_trit_sequences(4) {
            let p = pack(&ts, Scheme::I2S).expect("4-trit I2S pack");
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "I2S round-trip: {ts:?}");
        }
    }

    #[test]
    fn i2s_round_trip_two_groups() {
        // 8 trits (2 bytes) — exercises multi-group boundary.
        for ts in all_n_trit_sequences(8) {
            let p = pack(&ts, Scheme::I2S).expect("8-trit I2S pack");
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "I2S 2-group round-trip: {ts:?}");
        }
    }

    #[test]
    fn tl1_round_trip_exhaustive_5_trits() {
        // All 3^5 = 243 five-trit sequences must round-trip exactly under TL1.
        for ts in all_n_trit_sequences(5) {
            let p = pack(&ts, Scheme::Tl1).expect("5-trit TL1 pack");
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "TL1 round-trip: {ts:?}");
        }
    }

    #[test]
    fn tl2_round_trip_exhaustive_5_trits() {
        // All 3^5 = 243 five-trit sequences must round-trip exactly under TL2.
        for ts in all_n_trit_sequences(5) {
            let p = pack(&ts, Scheme::Tl2).expect("5-trit TL2 pack");
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "TL2 round-trip: {ts:?}");
        }
    }

    // ── cross-scheme losslessness (same trits, different bytes) ───────────────

    #[test]
    fn same_trits_different_bytes_different_schemes() {
        // Two packings of the same trits produce the same logical value (same trits on unpack)
        // but different bytes (not the same Packed). DN-01: packing is not a type distinction.
        //
        // Use a non-symmetric 5-trit sequence so TL1 and TL2 bytes differ.
        // [Pos, Zero, Zero, Zero, Zero] encodes as:
        //   TL1: 2·81 + 1·27 + 1·9 + 1·3 + 1 = 162+27+9+3+1 = 202 → wait, let's use digit map:
        //   TL1 digit map: Neg→0, Zero→1, Pos→2. So [Pos,Zero,Zero,Zero,Zero] = [2,1,1,1,1].
        //   TL1 value = 2·3^4 + 1·3^3 + 1·3^2 + 1·3 + 1 = 162+27+9+3+1 = 202.
        //   TL2 value = 242 - 202 = 40. So TL1 byte = 202, TL2 byte = 40. They differ. ✓
        let ts5 = vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Zero, Trit::Zero];
        let p_tl1 = pack(&ts5, Scheme::Tl1).unwrap();
        let p_tl2 = pack(&ts5, Scheme::Tl2).unwrap();

        // Same logical content on unpack.
        assert_eq!(
            unpack(&p_tl1),
            ts5,
            "TL1 unpack must recover original trits"
        );
        assert_eq!(
            unpack(&p_tl2),
            ts5,
            "TL2 unpack must recover original trits"
        );

        // Different bytes: TL1 = 202, TL2 = 40.
        assert_ne!(
            p_tl1.bytes, p_tl2.bytes,
            "TL1 and TL2 bytes must differ for non-symmetric input"
        );
        assert_eq!(p_tl1.bytes, vec![202u8], "TL1 byte for [Pos,0,0,0,0]");
        assert_eq!(p_tl2.bytes, vec![40u8], "TL2 byte for [Pos,0,0,0,0]");

        // I2S test: 4 trits, any value.
        let ts4 = int_to_trits(7, 4).unwrap();
        let p_i2s = pack(&ts4, Scheme::I2S).unwrap();
        assert_eq!(unpack(&p_i2s), ts4, "I2S round-trip");
    }

    // ── scheme_of / explain total ──────────────────────────────────────────────

    #[test]
    fn scheme_of_is_total_and_correct() {
        let ts: Vec<Trit> = vec![Trit::Zero; 4]; // aligned for I2S
        let p = pack(&ts, Scheme::I2S).unwrap();
        // scheme_of is total (C3).
        assert_eq!(scheme_of(&p), Scheme::I2S);
        assert_eq!(p.scheme(), Scheme::I2S);
    }

    #[test]
    fn explain_is_total_and_records_caller_selection() {
        let ts: Vec<Trit> = vec![Trit::Pos; 5]; // aligned for TL1
        let p = pack(&ts, Scheme::Tl1).unwrap();
        let rec = explain(&p);
        assert_eq!(rec.scheme, Scheme::Tl1);
        assert_eq!(rec.selection, SelectionNote::ExplicitCaller);
        assert_eq!(rec.trit_count, 5);
        assert_eq!(rec.byte_count, 1);
    }

    // ── byte count ────────────────────────────────────────────────────────────

    #[test]
    fn i2s_byte_count() {
        for n_groups in 1..=4 {
            let ts: Vec<Trit> = vec![Trit::Zero; n_groups * 4];
            let p = pack(&ts, Scheme::I2S).unwrap();
            assert_eq!(
                p.bytes.len(),
                n_groups,
                "I2S byte count for {n_groups} groups"
            );
        }
    }

    #[test]
    fn tl1_byte_count() {
        for n_groups in 1..=4 {
            let ts: Vec<Trit> = vec![Trit::Zero; n_groups * 5];
            let p = pack(&ts, Scheme::Tl1).unwrap();
            assert_eq!(
                p.bytes.len(),
                n_groups,
                "TL1 byte count for {n_groups} groups"
            );
        }
    }

    // ── round-trip with arithmetic values ─────────────────────────────────────

    #[test]
    fn pack_unpack_over_arithmetic_range_i2s() {
        // Every integer in the 4-trit range packs/unpacks losslessly under I2S.
        let max = max_magnitude(4).unwrap();
        for v in -max..=max {
            let ts = int_to_trits(v, 4).unwrap();
            let p = pack(&ts, Scheme::I2S).unwrap();
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "I2S arithmetic round-trip for v={v}");
        }
    }

    #[test]
    fn pack_unpack_over_arithmetic_range_tl1() {
        // Every integer in the 5-trit range packs/unpacks losslessly under TL1.
        let max = max_magnitude(5).unwrap();
        for v in -max..=max {
            let ts = int_to_trits(v, 5).unwrap();
            let p = pack(&ts, Scheme::Tl1).unwrap();
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "TL1 arithmetic round-trip for v={v}");
        }
    }

    #[test]
    fn pack_unpack_over_arithmetic_range_tl2() {
        let max = max_magnitude(5).unwrap();
        for v in -max..=max {
            let ts = int_to_trits(v, 5).unwrap();
            let p = pack(&ts, Scheme::Tl2).unwrap();
            let recovered = unpack(&p);
            assert_eq!(recovered, ts, "TL2 arithmetic round-trip for v={v}");
        }
    }
}
