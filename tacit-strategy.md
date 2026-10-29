## T-10: pre-read (ZIP arrives)
- Set up the environment and run the tests right away, so you know where you stand.
- Read the README twice. For each function to implement, note its **inputs, outputs, shapes and what each axis means**.
- Skim every provided helper (loaders, plotting). Anything they've written is something you don't have to.
- Write a 4–6 step plan with a target time for each step, and draft your questions.

## 0–5 min: clarify, then play back
Ask the 3–4 questions that would change what you build:
- **Priority:** "Should I aim to cover every part, or go deep on one?"
- **Data:** sampling rate and units; what one row or sample is; where t = 0 is; any known quality issues.
- **Goal:** "What generalization matters: new session, new subject, real time? Which metric do you care about?"
## 5–10 min: inspect (2–3 minutes, and say what you find)
- Shapes and what each axis means.
- Label set per group (watch for case or typo mismatches, and class imbalance).
- Missing values.
- Per-channel scale per group (dead or noisy channels).
- One plot.

Whatever you find becomes material for the Q&A.

## 10–40 min: build the thin slice
- **One function at a time.** Placeholders must always import. Run the tests after each function.
- **Know your shapes:** write each array's axes down and specify the axis explicitly every time. Convert ms to samples with rounding, as integers.
- **Unspecified parameter?** Choose it from domain knowledge, say why, and run one quick sanity check.
- **AI:** give it a spec (paste the docstring), read every line, and say aloud that you're using it. Never use it for decisions, and never give it context you haven't checked.

**Clock checkpoints:**

| Minute | Should have |
|---|---|
| 15 | Data loaded and inspected; first function passing |
| 25 | **An end-to-end number exists**, however crude |
| 35 | A sound evaluation (baseline, correct split) plus one improvement |
| 40 | Stop adding things. Clean up, and draft a 3-line summary: results, limitations, next steps |

## Evaluation principles
- Split by the unit you'll deploy across: session, subject, or time block.
- Anything fit on data is fit on the training fold only.
- Always compare against a baseline. Use a metric that's fair under class imbalance.
- A suspiciously good score means suspect leakage first.
- Report what you dropped, and why.

## 45–60 min: review and Q&A (have these ready)
- **What I did and why**, in 3 sentences.
- **What the numbers mean**, compared with the baseline, and why the two settings differ.
- **Limitations and failure modes:** data quality, assumptions, leakage risks.
- **With more time**, I'd do these 3 things.
- **Where I used AI** and how I verified it.

Expect "why this method?", "how do you know it's correct?", "what changes in real time?", and "what alternatives are there?"
