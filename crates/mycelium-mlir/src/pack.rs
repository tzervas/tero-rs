//! The **substrate byte-layout codec** for ternary values (RFC-0004 §5; DN-01) — the AOT/compiled
//! path's model of *how trits are physically packed into bytes* under a chosen [`PackScheme`].
//!
//! This is the substrate-level detail the Core IR deliberately omits (the type is packing-agnostic;
//! DN-01 §6) and the trusted kernel never carries (KC-3 — it lives here, in the AOT crate, not in
//! `mycelium-core`). Each scheme is a **bijective trit↔byte encoding**; decoding bytes under the
//! *wrong* scheme produces a different trit sequence — which is exactly the
//! MLIR-`transpose`/Rust-`packed` class of "a wrong layout tag misreads memory" bug DN-01 §4 cites,
//! and the soundness hazard the E3 differential (M-251) must catch (RFC-0004 §8; NFR-7).
//!
//! Schemes (RFC-0004 §5; the bitnet.cpp set + the two reference packings):
//! - `I2_S`, `TL1`, `TwoBitPerTrit` — **2 bits/trit**, 4 trits/byte, distinguished by their code
//!   LUT (the three rotations of `{0,1,2}`), so the same trits pack to *different* bytes.
//! - `FiveTritPerByte` — the **base-3 reference** packing, 5 trits/byte (`3⁵ = 243 ≤ 256`; **1.6
//!   b/w**), the near-optimal-density encoding (entropy limit `log₂3 ≈ 1.585`).
//! - `TL2` — the **true bitnet.cpp TL2 layout**: 3 trits → a **5-bit LUT index** (`c = d₀ + 3·d₁ +
//!   9·d₂ ∈ [0,27)`), bit-packed as a contiguous 5-bit-field stream ⇒ **1.67 b/w** (`5/3`). It is
//!   *less* dense than `FiveTritPerByte` on purpose: the 5-bit index is directly LUT-addressable
//!   (the "TL" = ternary lookup), trading a little density for fast decode.
//! - `Unpacked` — 1 trit/byte.
//!
//! **A5-08 — resolved (M-360 real-layout increment).** `TL2` now realizes the published bitnet.cpp
//! **1.67 b/w** (3-trits-per-5-bits LUT-index bitstream), matching the selector's cost model
//! (`packing_bits_per_element(Tl2) = 1.67` in `mycelium-select`) — the prior 1.6-b/w base-3
//! placeholder (which shared `FiveTritPerByte`'s layout) is retired; the two schemes are now
//! genuinely distinct densities. The M-360 native TL2 **dot kernel** (`bitnet`) decodes this layout.
//! **Honest scope (VR-5):** this realizes the bitnet.cpp TL2 *density and 5-bit-LUT-index semantics*
//! (3 trits → a 5-bit code), bit-packed contiguously; the exact upstream *byte/bit ordering* of
//! bitnet.cpp's internal buffer is not claimed byte-identical (that needs the upstream source to
//! verify) — our codec is self-consistent (round-trip identity) and oracle-checked, which is what
//! the value semantics and the differential require.
//!
//! Decoding is **total** (never panics): an out-of-range code/byte folds `mod 3`, so reading a
//! buffer under a mismatched scheme yields *some* trit sequence deterministically — a misread, not
//! a crash. Round-trip under the *same* scheme is the identity ([`pack_trits`] ∘ [`unpack_trits`]).
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use mycelium_core::{PackScheme, Trit};

/// A packing-codec error. A short buffer is **explicit** (A5-03): `unpack_trits` never silently
/// truncates to fewer trits than requested — a buffer that cannot hold `count` trits under the
/// scheme's density is the diagnostic [`PackError::BufferTooShort`], not a quiet partial decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    /// The byte buffer is too short to decode `count` trits under `scheme`: `count` needs at least
    /// `needed` bytes ([`needed_bytes`]) but only `got` were supplied.
    BufferTooShort {
        /// The trit count requested.
        count: usize,
        /// The minimum bytes the scheme requires for `count` trits.
        needed: usize,
        /// The bytes actually supplied.
        got: usize,
    },
}

impl core::fmt::Display for PackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackError::BufferTooShort { count, needed, got } => write!(
                f,
                "buffer too short to decode {count} trits: need {needed} bytes, got {got}"
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// Trits-per-byte for the **byte-aligned** schemes (the packing density's structural form). `TL2` is
/// *not* byte-aligned (it is a 5-bit-field bitstream — see [`needed_bytes`]); callers must route TL2
/// through the bitstream path, not this. `FiveTritPerByte` remains the byte-aligned 5-trits/byte
/// base-3 reference.
fn group_size(scheme: PackScheme) -> usize {
    match scheme {
        PackScheme::Unpacked => 1,
        PackScheme::TwoBitPerTrit | PackScheme::I2S | PackScheme::Tl1 => 4,
        PackScheme::FiveTritPerByte => 5,
        // TL2 is a bitstream; this is the byte-aligned fallback used only by the misread model.
        PackScheme::Tl2 => 5,
    }
}

/// The true bitnet.cpp **TL2** layout: 3 trits → one 5-bit LUT-index code (`3⁵ = 243`, but `3³ = 27`
/// fits a 5-bit field), bit-packed contiguously ⇒ `5/3 ≈ 1.67` b/w.
const TL2_TRITS_PER_GROUP: usize = 3;
const TL2_BITS_PER_GROUP: usize = 5;

/// Bytes required to hold `count` trits under `scheme` — the buffer-bound model. For the byte-aligned
/// schemes this is `count.div_ceil(trits_per_byte)`; for the bitstream `TL2` it is the packed
/// 5-bit-code stream length (`⌈5·⌈count/3⌉ / 8⌉`).
#[must_use]
pub fn needed_bytes(scheme: PackScheme, count: usize) -> usize {
    match scheme {
        PackScheme::Tl2 => {
            let groups = count.div_ceil(TL2_TRITS_PER_GROUP);
            (TL2_BITS_PER_GROUP * groups).div_ceil(8)
        }
        _ => count.div_ceil(group_size(scheme)),
    }
}

/// Write the low `TL2_BITS_PER_GROUP` bits of `code` into `buf` starting at bit offset `bit_off`
/// (LSB-first, little-endian within the stream). `buf` must be long enough ([`needed_bytes`] sizes it).
fn write_tl2_code(buf: &mut [u8], bit_off: usize, code: u8) {
    for b in 0..TL2_BITS_PER_GROUP {
        if (code >> b) & 1 == 1 {
            buf[(bit_off + b) / 8] |= 1 << ((bit_off + b) % 8);
        }
    }
}

/// Read the 5-bit TL2 code at bit offset `bit_off` (may straddle a byte boundary). Out-of-range bytes
/// read as 0 (total, like the byte-aligned misread) — but [`needed_bytes`] guarantees the bits exist
/// for an in-bounds decode.
fn read_tl2_code(bytes: &[u8], bit_off: usize) -> u8 {
    let byte = bit_off / 8;
    let shift = bit_off % 8;
    let lo = u16::from(bytes.get(byte).copied().unwrap_or(0));
    let hi = u16::from(bytes.get(byte + 1).copied().unwrap_or(0));
    let window = (lo | (hi << 8)) >> shift;
    #[allow(clippy::cast_possible_truncation)]
    {
        (window & 0x1F) as u8
    }
}

/// Pack `trits` under the true TL2 layout (3 trits → 5-bit code, bit-packed). Bijective with
/// [`unpack_tl2`].
fn pack_tl2(trits: &[Trit]) -> Vec<u8> {
    let mut bytes = vec![0u8; needed_bytes(PackScheme::Tl2, trits.len())];
    for (gi, chunk) in trits.chunks(TL2_TRITS_PER_GROUP).enumerate() {
        // code = d₀ + 3·d₁ + 9·d₂, each dₖ = d01(trit) ∈ {0,1,2} ⇒ code ∈ [0, 27).
        let mut code: u8 = 0;
        let mut p: u8 = 1;
        for &t in chunk {
            code += d01(t) * p;
            p *= 3;
        }
        write_tl2_code(&mut bytes, gi * TL2_BITS_PER_GROUP, code);
    }
    bytes
}

/// Decode `count` trits from a TL2 bitstream. Each trit `i` is digit `p = i % 3` of the 5-bit code at
/// group `g = i / 3`: `digit = (code / 3ᵖ) mod 3`.
fn unpack_tl2(bytes: &[u8], count: usize) -> Vec<Trit> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let g = i / TL2_TRITS_PER_GROUP;
        let p = i % TL2_TRITS_PER_GROUP;
        let code = read_tl2_code(bytes, g * TL2_BITS_PER_GROUP);
        let digit = (code / 3u8.pow(u32::try_from(p).expect("p < 3"))) % 3;
        out.push(from_d01(digit));
    }
    out
}

/// A trit as a base-3 digit `{0, 1, 2}` (`Neg→0, Zero→1, Pos→2`).
fn d01(t: Trit) -> u8 {
    match t {
        Trit::Neg => 0,
        Trit::Zero => 1,
        Trit::Pos => 2,
    }
}

/// The inverse of [`d01`], total via `mod 3` (so a cross-scheme misread decodes to *a* trit, never
/// a panic).
fn from_d01(d: u8) -> Trit {
    match d % 3 {
        0 => Trit::Neg,
        1 => Trit::Zero,
        _ => Trit::Pos,
    }
}

/// The per-scheme 2-bit code LUT — the three rotations of `{0,1,2}`, so `I2_S`/`TL1`/`TwoBitPerTrit`
/// pack identical trits to *different* bytes (the distinguishing detail E3 relies on).
fn two_bit_rot(scheme: PackScheme) -> u8 {
    match scheme {
        PackScheme::I2S => 0,
        PackScheme::Tl1 => 2,
        PackScheme::TwoBitPerTrit => 1,
        _ => 0,
    }
}

/// The base-3 digit order for the 5-trit-per-byte schemes — `TL2` keeps `d01`, `FiveTritPerByte`
/// reverses it, so they remain distinct encodings.
fn base3_reversed(scheme: PackScheme) -> bool {
    matches!(scheme, PackScheme::FiveTritPerByte)
}

/// Encode `trits` to bytes under `scheme` (bijective; the AOT path's physical buffer). The final
/// partial group is zero-padded; [`unpack_trits`] reads exactly the requested count back.
#[must_use]
pub fn pack_trits(trits: &[Trit], scheme: PackScheme) -> Vec<u8> {
    if matches!(scheme, PackScheme::Tl2) {
        return pack_tl2(trits); // the bitstream layout, not byte-aligned
    }
    let g = group_size(scheme);
    let mut bytes = Vec::with_capacity(trits.len().div_ceil(g));
    for chunk in trits.chunks(g) {
        let byte = match scheme {
            PackScheme::Unpacked => d01(chunk[0]),
            PackScheme::TwoBitPerTrit | PackScheme::I2S | PackScheme::Tl1 => {
                let rot = two_bit_rot(scheme);
                let mut b: u8 = 0;
                for (i, &t) in chunk.iter().enumerate() {
                    let code = (d01(t) + rot) % 3; // ∈ {0,1,2}, fits 2 bits
                    b |= code << (2 * i);
                }
                b
            }
            PackScheme::FiveTritPerByte | PackScheme::Tl2 => {
                let rev = base3_reversed(scheme);
                let mut b: u16 = 0;
                let mut p: u16 = 1;
                for &t in chunk {
                    let digit = if rev { 2 - d01(t) } else { d01(t) };
                    b += u16::from(digit) * p;
                    p *= 3;
                }
                u8::try_from(b).expect("five base-3 digits fit in a byte (3^5 = 243 < 256)")
            }
        };
        bytes.push(byte);
    }
    bytes
}

/// Decode `count` trits from `bytes` under `scheme`. A code/byte outside the scheme's valid range
/// folds `mod 3`, so reading a buffer packed under a *different* scheme yields a deterministic
/// (wrong) trit sequence — the misread, never a panic.
///
/// A buffer too short for `count` trits is the explicit [`PackError::BufferTooShort`] (A5-03):
/// the codec never silently returns fewer trits than requested. When the buffer is long enough,
/// decoding cannot fail.
pub fn unpack_trits(
    bytes: &[u8],
    scheme: PackScheme,
    count: usize,
) -> Result<Vec<Trit>, PackError> {
    let needed = needed_bytes(scheme, count);
    if bytes.len() < needed {
        return Err(PackError::BufferTooShort {
            count,
            needed,
            got: bytes.len(),
        });
    }
    if matches!(scheme, PackScheme::Tl2) {
        return Ok(unpack_tl2(bytes, count)); // bitstream decode
    }
    let g = group_size(scheme);
    let mut out = Vec::with_capacity(count);
    'outer: for (bi, &byte) in bytes.iter().enumerate() {
        for i in 0..g {
            if bi * g + i >= count {
                break 'outer;
            }
            let trit = match scheme {
                PackScheme::Unpacked => from_d01(byte),
                PackScheme::TwoBitPerTrit | PackScheme::I2S | PackScheme::Tl1 => {
                    let rot = two_bit_rot(scheme);
                    let code = (byte >> (2 * i)) & 0b11;
                    // invert the rotation: d01 = code - rot (mod 3); +3 keeps it non-negative.
                    from_d01((code + 3 - rot) % 3)
                }
                PackScheme::FiveTritPerByte | PackScheme::Tl2 => {
                    let rev = base3_reversed(scheme);
                    let digit = (byte / 3u8.pow(u32::try_from(i).expect("i < 5"))) % 3;
                    from_d01(if rev { 2 - digit } else { digit })
                }
            };
            out.push(trit);
        }
    }
    Ok(out)
}

/// Re-materialize trits through a pack-then-read round-trip where the buffer is **packed as**
/// `packed_as` but **read back as** `read_as` (the recorded `Meta.physical` tag). When the tag is
/// correct (`packed_as == read_as`) this is the identity; a wrong tag *misreads* the buffer — the
/// soundness hazard the E3 differential catches (RFC-0004 §8; NFR-7).
#[must_use]
pub fn relayout_trits(trits: &[Trit], packed_as: PackScheme, read_as: PackScheme) -> Vec<Trit> {
    let mut bytes = pack_trits(trits, packed_as);
    // A denser `packed_as` emits fewer bytes than a sparser `read_as` needs for the same count;
    // zero-pad to the bytes `read_as` requires so the read is the modeled misread (a wrong layout
    // tag over the *same* buffer, zero-extended) — never an explicit short.
    let needed = needed_bytes(read_as, trits.len());
    if bytes.len() < needed {
        bytes.resize(needed, 0);
    }
    unpack_trits(&bytes, read_as, trits.len())
        .expect("buffer zero-padded to read_as's required length, so the read cannot be short")
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SCHEMES: [PackScheme; 6] = [
        PackScheme::Unpacked,
        PackScheme::TwoBitPerTrit,
        PackScheme::FiveTritPerByte,
        PackScheme::I2S,
        PackScheme::Tl1,
        PackScheme::Tl2,
    ];

    fn sample() -> Vec<Trit> {
        // 11 trits: spans a partial 4-group and a partial 5-group, mixed values.
        vec![
            Trit::Neg,
            Trit::Pos,
            Trit::Zero,
            Trit::Pos,
            Trit::Neg,
            Trit::Zero,
            Trit::Pos,
            Trit::Pos,
            Trit::Neg,
            Trit::Zero,
            Trit::Neg,
        ]
    }

    #[test]
    fn round_trip_is_identity_under_the_same_scheme() {
        for s in ALL_SCHEMES {
            let t = sample();
            let back = unpack_trits(&pack_trits(&t, s), s, t.len()).unwrap();
            assert_eq!(back, t, "scheme {s:?} must round-trip losslessly");
            // relayout with a matching tag is the identity.
            assert_eq!(relayout_trits(&t, s, s), t);
        }
    }

    #[test]
    fn the_three_bitnet_schemes_are_mutually_distinct_encodings() {
        // The E3 precondition: a buffer packed under one bitnet scheme, read under another, misreads.
        let bitnet = [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2];
        let t = sample();
        for &a in &bitnet {
            for &b in &bitnet {
                if a != b {
                    assert_ne!(
                        relayout_trits(&t, a, b),
                        t,
                        "packing as {a:?} then reading as {b:?} must diverge"
                    );
                }
            }
        }
    }

    #[test]
    fn an_all_zero_buffer_still_misreads_across_schemes() {
        // Even the degenerate all-Zero value diverges (the LUTs map Zero differently), so E3 does
        // not rely on lucky test data.
        let t = vec![Trit::Zero; 5];
        assert_ne!(relayout_trits(&t, PackScheme::I2S, PackScheme::Tl1), t);
        assert_ne!(relayout_trits(&t, PackScheme::I2S, PackScheme::Tl2), t);
    }

    #[test]
    fn decoding_is_total_on_arbitrary_bytes() {
        // Reading arbitrary bytes (e.g. a TL2 buffer under a 2-bit scheme) never panics, as long as
        // the buffer is long enough for the requested count. The sparsest scheme is `Unpacked` at
        // 1 trit/byte, so 5 bytes supply at least 5 trits under *every* scheme; request 5 so the
        // length precondition holds across the board (A5-03 makes a short buffer an explicit error,
        // not a panic or a silent truncation — exercised separately below).
        let bytes = [0xFF, 0x00, 0xAB, 0x7C, 242];
        for s in ALL_SCHEMES {
            let _ = unpack_trits(&bytes, s, 5).unwrap();
        }
    }

    #[test]
    fn tl2_realizes_the_true_167_bits_per_weight() {
        // A5-08 closure: TL2 is the true bitnet.cpp 1.67-b/w layout (3 trits → 5 bits), strictly
        // *less* dense than the FiveTritPerByte base-3 reference (1.6 b/w, 5 trits/byte). The two are
        // now distinct densities — TL2 uses more bytes for the same count.
        for &count in &[3usize, 6, 24, 100, 1000, 4096] {
            let tl2 = needed_bytes(PackScheme::Tl2, count);
            let five = needed_bytes(PackScheme::FiveTritPerByte, count);
            // 1.67 b/w ⇒ bytes ≈ count·5/3/8; check it matches the exact bitstream length and that
            // it exceeds (or equals, only for the tiniest) the 1.6-b/w reference.
            assert_eq!(
                tl2,
                (5 * count.div_ceil(3)).div_ceil(8),
                "TL2 bitstream length at {count}"
            );
            assert!(
                tl2 >= five,
                "1.67 b/w must not be denser than 1.6 b/w at {count}"
            );
        }
        // Observed b/w over a large buffer is ~1.667 (5/3), distinctly above FiveTritPerByte's ~1.6.
        let n = 30_000;
        #[allow(clippy::cast_precision_loss)]
        let bpw = (needed_bytes(PackScheme::Tl2, n) as f64) * 8.0 / (n as f64);
        assert!(
            (1.66..=1.68).contains(&bpw),
            "TL2 b/w {bpw} should be ≈1.67"
        );
    }

    #[test]
    fn tl2_and_five_trit_per_byte_are_now_distinct_layouts() {
        // Before A5-08 closure TL2 shared FiveTritPerByte's byte-aligned base-3 layout (1.6 b/w); now
        // TL2 is the 1.67-b/w bitstream, so packing as one and reading as the other misreads.
        let t = sample();
        assert_ne!(
            relayout_trits(&t, PackScheme::Tl2, PackScheme::FiveTritPerByte),
            t,
            "TL2 packed, read as the base-3 reference, must misread"
        );
        assert_ne!(
            relayout_trits(&t, PackScheme::FiveTritPerByte, PackScheme::Tl2),
            t
        );
    }

    #[test]
    fn a_short_buffer_is_an_explicit_error_not_a_silent_truncation() {
        // A5-03 mutant-witness: before the fix `unpack_trits` silently returned fewer trits than
        // requested when `bytes` was too short. Now it is an explicit `BufferTooShort`.
        // I2S packs 4 trits/byte: 1 byte holds at most 4 trits, so asking for 5 must refuse.
        assert_eq!(
            unpack_trits(&[0u8], PackScheme::I2S, 5),
            Err(PackError::BufferTooShort {
                count: 5,
                needed: 2,
                got: 1,
            })
        );
        // An empty buffer cannot supply even one trit.
        assert_eq!(
            unpack_trits(&[], PackScheme::Tl2, 1),
            Err(PackError::BufferTooShort {
                count: 1,
                needed: 1,
                got: 0,
            })
        );
        // The exact-fit boundary succeeds (no off-by-one refusal): 2 bytes hold 8 trits under I2S.
        assert!(unpack_trits(&[0u8, 0u8], PackScheme::I2S, 8).is_ok());
    }
}
