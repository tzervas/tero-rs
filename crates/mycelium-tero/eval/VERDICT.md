# Layer-2 eval gate — VERDICT (append-only)

The M-1018 gate record (DN-87 §6.1). Each run appends a section; history is never rewritten. The gate is **Closed by default** and opens only on a measured Layer-2 win that keeps provenance at 1.0 and latency within the band. A **Closed gate is the honest, expected outcome** for this ~5k-row structured corpus — the system serves Layer-1 answers and the improved-on-RAG claim stays aspiration (G2/VR-5).

## Run 1 — gate CLOSED (serving Layer-1)

- host: x86_64-linux, 4 hw threads
- seed (Layer-2 master): 0x7E7010185EEDC0DE
- questions: 16 · k = 5 · codebook = 5141 records (0 refused, never-silent)
- correctness@1: Layer-1 10/16 (0.625) · Layer-2 6/16 (0.375)
- correctness@5: Layer-1 16/16 (1.000) · Layer-2 8/16 (0.500)
- provenance fidelity: Layer-1 1.000 · Layer-2 1.000 (must be 1.0 to open)
- latency (Empirical, ns/query, 5 trials): Layer-1 3246771 · Layer-2 85884490
- verdict: CLOSED (serving Layer-1) — Layer-2 correctness@1 0.3750 did not beat Layer-1 0.6250 beyond the 20% band; Layer-2 latency 85884490ns/query exceeds Layer-1 3246771ns/query beyond the 20% band
