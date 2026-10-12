**Week 1 — Python Pipeline** _(complete)_
- Project structure, README, data contract
- Env setup, CMake + libtorch verified
- Willett `.mat` loading → `[T, 256]` float32, 8800 trials
- `BlockZScorer`: per-channel per-block z-score → `constants.json`
- `vocab.py`: ARPAbet + cmudict + contractions
- `SpeechBCIModel`: 3-layer bidirectional GRU, hidden 1024, 45.7M params
- Training loop (AdamW, CTCLoss, CPU), Run 1 + Run 2 (ReduceLROnPlateau)
- `export.py` + `verify_export.py`, parity at `atol=1e-5`
- `artifacts/` locked: `model.pt`, `model_config.json`, `constants.json`

**Week 2 — C++ Core Components** _(complete)_
- C++ fundamentals learned through building: headers, namespaces, `const`, `struct`, `IValue`, `j.at()` convention
- `ModelRunner`: loads `model.pt`, validates config invariants, `forward` verified `[25,256]→[25,40]`
- `LatencyTimer`: named-scope, p50/p95/p99 JSON report — inference p50 ≈ 5.07ms
- `ZScoreConstants`: loads `constants.json`, raw-pointer active block, 10/10 GTest passing, Python parity at `atol=1e-5`
- Google Test wired via FetchContent, `ndr_runtime` static lib established

**Week 3 — Batch Mode, End to End** _(current — not started)_
- CTC greedy decoder in C++: `[T,40]` logits → phoneme sequence
- Full pipeline in `main.cpp`: `constants.json` → normalize → `forward` → greedy decode → phoneme string
- Parity check on the _decoded phoneme output_ (Python vs C++), not just raw tensors
- Design log: batch-mode-complete-before-streaming, citing the "batch is streaming with known boundaries" principle from the data contract

**Week 4 — Real Training Run**

- W&B logging, proper hyperparameter sweep
- Exit criterion: loss curve decreasing, greedy WER on dev set is a real number
- Re-export best checkpoint, re-verify parity end to end (including decoded output, per Week 3's new parity bar)

**Week 5 — Beam Search + KenLM**

- KenLM build on Apple Silicon (budget extra time)
- 4-gram LM (LibriSpeech or Willett transcript corpus)
- Beam search decoder in C++ replacing greedy
- WER delta documented: greedy vs. beam vs. beam+LM
- Design log note: KenLM chosen as the working baseline; flashlight-text comparison stays open, revisited at Week 8–9

**Week 6 — Streaming Core**

- `forward_streaming(x, h)` added to Python model, re-exported, parity-verified
- Rolling circular buffer `[context_bins, C]`
- Activity-threshold boundary detection
- Hidden state: carried within utterance, hard reset at boundary
- Greedy partial decode on every stride
- Design log: stateful vs. stateless forward decision

**Week 7 — Streaming + Beam Search at Boundary**

- Full beam search + LM fires at detected boundary
- End-to-end streaming test: bins fed one at a time → partials → final at boundary
- Latency measured specifically in streaming mode
- Test cases: hidden-state reset, boundary edge cases

**Week 8 — Server + pybind11 + Decode-Layer Decision**

- POST `/decode` batch endpoint (crow or uWebSockets)
- WebSocket `/stream` endpoint
- pybind11 bindings
- KenLM vs. `flashlight-text` `LexiconDecoder` evaluated here (CMake dependency, not file-copying); decision + rationale go in design log
- LSL vs. WebSocket noted in README now if not already; real decision only if server layer forces it

**Week 9 — EvalAI Submission**

- Final checkpoint selected
- Batch evaluation on Brain-to-Text 2024
- EvalAI leaderboard submission
- WER documented, latency table complete (add Python inference baseline row per earlier review feedback)

**Week 10 — Polish**

- Structured JSON logging per request
- Graceful error handling throughout
- Build portability: clean clone → working binary in under 30 minutes
- All GTest + pytest passing
- README quickstart elevated (per review feedback), architecture diagram
- Qualify the "preprocessing agnostic" claim in README (per review feedback)

**Week 11 — Delivery**

- Final leaderboard entry
- Repo polish: commit history, badges, benchmark table
- 2–4 page technical writeup: problem, architecture, decisions, results