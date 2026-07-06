# Provenance Guard — System Specification

## 1. Detection Signals

### Signal 1: Statistical Text Analysis (local, no API)

**What it measures:** Three structural properties of text that differ between human and AI writing.

**Sub-signals:**

| Sub-signal | What it measures | How it's computed | Output | Why it matters |
|---|---|---|---|---|
| Burstiness | Sentence length variance | Coefficient of variation (std_dev / mean) of word counts per sentence | 0.0–1.0 float (0 = high variance/human-like, 1 = uniform/AI-like) | Humans write "I ran. Then I stopped at the old wooden fence and stared for what felt like hours." — 2 words then 16. AI tends toward 10, 12, 11, 13. CV > 0.6 → strongly human. CV < 0.3 → AI-like. Score = `clamp(1.0 - cv/0.8, 0, 1)` |
| Filler phrases | AI-characteristic transition/hedge words | Count occurrences of 20 known AI filler phrases ("it is important to note," "furthermore," "in conclusion," etc.) divided by sentence count | 0.0–1.0 float (0 = no filler/human, 1 = heavy filler/AI) | AI text uses formulaic transitions ("Furthermore," "Additionally," "It should be noted") at much higher rates than human creative writing. Score = `clamp(filler_density / 1.5, 0, 1)` where filler_density = filler_hits / sentence_count |
| Structural patterns | Sentence-start diversity and length uniformity | Combines two measures: (1) length uniformity = `1.0 - min(1.0, cv)`, and (2) sentence-start repetition = `1.0 - unique_starts / total_starts`. Averaged 50/50. | 0.0–1.0 float (0 = varied structure/human, 1 = uniform structure/AI) | AI text often starts sentences with the same patterns ("The," "It," "This") and maintains consistent sentence lengths. Human writing varies both more naturally. |

**Combined signal output:** Weighted average → single float 0.0–1.0.
```
statistical_score = burstiness * 0.35 + filler_phrases * 0.35 + structural_patterns * 0.3
```
Burstiness and filler phrases share the highest weight because they are the most reliable differentiators at typical text lengths (50–500 words). Structural patterns provide a supporting signal.

**Short text fallback:** If the text has fewer than 10 words or fewer than 2 sentences, the statistical signal returns 0.5 ("no information") with a detail note explaining it couldn't analyze the text. This prevents short poems or tweets from getting false statistical scores.

### Signal 2: LLM-Based Classification (Groq API)

**What it measures:** Semantic and stylistic patterns — "AI voice" — that statistical measures can't capture: hedging language ("It's important to note that..."), parallel list structures, absence of personal voice, and emotional flatness.

**How it works:** The text (truncated to 3,000 chars to fit context windows) is sent to Groq's LLaMA 3.1 8B model with a structured prompt asking it to score the text 0.0–1.0 and provide a one-sentence rationale.

**Output:** `{"score": 0.0-1.0, "rationale": "string"}`
- 0.0 = definitely human-written
- 1.0 = definitely AI-generated
- The rationale is stored in the audit log but not shown in the transparency label (it's too technical for end users).

**Failure mode:** If the Groq API is unreachable or returns unparseable JSON, the signal returns `{"score": 0.5, "rationale": "LLM analysis unavailable"}`. A score of 0.5 means "no information," which will pull the combined score toward uncertain — the safe default.

### How signals combine into a single confidence score

```
weighted_average = statistical_score * 0.4 + llm_score * 0.6
```

The LLM gets 60% weight because it captures semantic patterns (hedging, voice, structure) that pure statistics miss. The statistical signal gets 40% as a grounding check — it's deterministic and can't be confused by prompting artifacts.

**Disagreement adjustment:** When the two signals disagree by more than 0.3, the combined score is pulled toward 0.5:
```
disagreement = abs(stat_score - llm_score)
if disagreement > 0.3:
    pull_factor = min(1.0, (disagreement - 0.3) / 0.4)   # 0→1 as disagreement goes 0.3→0.7
    weighted = weighted + (0.5 - weighted) * pull_factor * 0.5
```

This is the key design decision for handling false positives. If one signal says "AI" and the other says "human," the system admits uncertainty rather than picking a side. The pull is proportional — a disagreement of 0.35 barely adjusts; a disagreement of 0.7 pulls hard toward 0.5.

---

## 2. Uncertainty Representation

### What scores mean — worked examples

| Statistical | LLM | Disagreement | Adjustment | Final Score | Label |
|---|---|---|---|---|---|
| 0.15 | 0.20 | 0.05 (agree) | none | 0.18 | Human (82% confidence) |
| 0.80 | 0.90 | 0.10 (agree) | none | 0.86 | AI (86% confidence) |
| 0.20 | 0.80 | 0.60 (strong disagree) | pulled toward 0.5 | 0.54 | Uncertain |
| 0.35 | 0.60 | 0.25 (mild disagree) | none | 0.50 | Uncertain |
| 0.40 | 0.85 | 0.45 (disagree) | moderate pull | 0.64 | Uncertain |
| 0.90 | 0.95 | 0.05 (agree) | none | 0.93 | AI (93% confidence) |

### What a score of 0.6 means to the system

A score of 0.6 falls in the uncertain band (0.25–0.75). It means: "Our signals lean slightly toward AI-generated, but not enough to make a confident claim." This could happen when:
- The LLM detects some AI-like patterns (score ~0.7) but the statistical analysis finds human-like burstiness (score ~0.4), and the disagreement adjustment pulled the result toward center.
- Both signals are mildly suspicious but neither is confident.

The user sees the uncertain label, which says the system can't determine with confidence — not "probably AI" or "slightly AI." The 0.25–0.75 band is deliberately wide because a false positive (accusing a human of using AI) is more harmful than a false negative (missing AI content). A narrower band like 0.4–0.6 would force the system to make confident calls it can't back up.

### Threshold boundaries

| Score Range | Attribution | Meaning |
|---|---|---|
| 0.00–0.25 | `human_written` | Both signals agree the text shows strong human characteristics |
| 0.26–0.74 | `uncertain` | Signals disagree, are ambiguous, or the text is too short to analyze reliably |
| 0.75–1.00 | `ai_generated` | Both signals agree the text shows strong AI characteristics |

The thresholds are asymmetric in practice: the disagreement adjustment makes it hard to reach 0.75+ unless both signals agree, so the system rarely makes a confident "AI" call unless the evidence is strong from multiple angles. This is intentional — the system is designed to be conservative about accusations.

---

## 3. Transparency Label Design

Three label variants, written for a non-technical reader on a creative writing platform:

### Variant 1: High-Confidence AI (score ≥ 0.75)

> **AI-Assisted Content** — This content has been flagged as likely AI-generated. Our analysis detected patterns consistent with AI writing tools. Confidence: 92%. If you believe this is incorrect, you can submit an appeal.

**Design choices:**
- "Flagged as likely" rather than "is" — communicates probability, not certainty.
- Includes the confidence percentage so the reader can gauge how sure the system is. 92% reads differently than 76%.
- Ends with the appeal path so the creator immediately sees recourse.
- Title says "AI-Assisted" not "AI-Generated" — gentler framing that acknowledges AI might be a tool, not the sole author.

### Variant 2: High-Confidence Human (score ≤ 0.25)

> **Original Work** — This content appears to be human-written. Our analysis found patterns consistent with original human authorship. Confidence: 85%.

**Design choices:**
- Title is "Original Work" — a positive framing that the creator can feel good about.
- "Appears to be" maintains epistemic honesty.
- Confidence is calculated as `(1 - score) * 100` — a score of 0.15 becomes "85% confidence in human authorship." This makes the percentage always represent confidence in the stated verdict, not a raw AI probability.
- No appeal link — there's nothing to contest.

### Variant 3: Uncertain (0.25 < score < 0.75)

> **Under Review** — We couldn't determine with confidence whether this content is AI-generated or human-written. This may reflect mixed authorship, an unusual writing style, or heavy editing. The creator can provide additional context through an appeal.

**Design choices:**
- Title is "Under Review" — neutral, doesn't stigmatize.
- Lists three legitimate, non-accusatory reasons for uncertainty: mixed authorship, unusual style, heavy editing. Each is a real scenario a human writer might find themselves in.
- Points the creator to the appeal system to provide context (e.g., "I have an unusual writing style" or "I edited this heavily from an AI draft").
- No percentage shown — displaying "52% AI" would mislead readers into treating it as meaningful precision when it really means "we don't know."

---

## 4. Appeals Workflow

### Who can appeal
Any user with the `content_id` of a prior submission. One appeal per submission — if an appeal is already under review, a second attempt is rejected with HTTP 409.

### What information they provide
`POST /appeal` with a JSON body:
```json
{
  "content_id": "aeb3ec8a-a6ab-4981-81b7-c77a9a75f979",
  "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."
}
```
The `creator_reasoning` field is free text. It should capture why the creator believes the classification is wrong. The appeal is stored alongside the original decision so a reviewer can evaluate both.

### What happens when an appeal is received

1. **Validation:** System checks that the submission exists and is not already under review.
2. **Status change:** `submission.status` changes from `"classified"` to `"under_review"`.
3. **Appeal record created:** Stored with: `appeal_id`, `content_id`, `creator_id`, `appeal_reasoning`, `original_attribution`, `original_confidence`, `status: "under_review"`, `created_at` timestamp.
4. **Audit log entry:** A new entry with `event_type: "appeal_submitted"` is appended, capturing the appeal ID, the original classification, confidence, and the creator's reasoning.
5. **Response to creator:** Confirmation message explaining the appeal was received and a moderator will review.

### What a human reviewer sees

When a reviewer opens `GET /status/{content_id}`, they see:

```json
{
  "content_id": "aeb3ec8a-...",
  "attribution": "ai_generated",
  "confidence_score": 0.839,
  "status": "under_review",
  "label": "This content has been flagged as likely AI-generated...",
  "appeals": [
    {
      "appeal_id": "f9d2640d-...",
      "content_id": "aeb3ec8a-...",
      "creator_id": "label-test-3",
      "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
      "original_attribution": "ai_generated",
      "original_confidence": 0.839,
      "status": "under_review",
      "created_at": "2026-07-06T01:21:22.047002+00:00"
    }
  ]
}
```

The reviewer sees the original verdict and confidence, the creator's reasoning, and can cross-reference the audit log to see both signal scores (statistical: 0.75, LLM: 0.90) — understanding that both signals agreed this looked AI-generated, but the creator claims non-native English writing style as the explanation.

---

## 5. Anticipated Edge Cases

### Edge Case 1: Intentionally repetitive poetry
A poet who writes in a minimalist style — short lines, repeated phrases as a literary device (think: "Do not go gentle into that good night"). The statistical signal's repetition sub-signal will score high (AI-like) because it can't distinguish intentional literary repetition from AI pattern repetition. The burstiness signal may also score high (AI-like) if the lines are uniform length.

**What the system does:** The LLM signal may correctly identify it as human poetry, creating signal disagreement. The disagreement adjustment pulls toward uncertain. The creator sees "Under Review" and can appeal with context about their writing style. This is the intended behavior — the system admits uncertainty rather than making a confident wrong call.

**What a better system would do:** Genre-aware analysis that adjusts expectations for poetry vs. prose. We don't have this; it's a known limitation.

### Edge Case 2: Human-written formulaic content
A technical writer creating product descriptions or FAQ entries. The content is factual, uses standard phrasing ("This product features..."), has uniform sentence lengths, and lacks personal voice. Every signal will lean toward AI-generated because the content genuinely looks like AI output — it's just written by a human following a template.

**What the system does:** Both signals may agree it looks AI-generated, producing a high confidence score and the "AI-Assisted Content" label. This is a false positive. The creator must appeal and explain their role.

**Why this is hard to fix:** The statistical properties of template-driven human writing and AI writing are genuinely similar. The system can't distinguish "writes like AI because they follow a template" from "is AI." This is an inherent limitation of content-only analysis — you'd need authorship metadata (keystroke patterns, edit history) to resolve it.

### Edge Case 3: AI-generated text that has been manually edited
A writer uses ChatGPT to generate a draft, then rewrites 40% of it — adding personal anecdotes, varying sentence structure, fixing the "AI voice." The statistical signal may show human-like burstiness (from the edits) while the LLM signal detects residual AI patterns in the untouched portions.

**What the system does:** The signals disagree, the disagreement adjustment activates, and the score lands in the uncertain band. The label says "Under Review" with no confident accusation. This is actually the correct answer — the content is mixed authorship, and the system communicates exactly that.

### Edge Case 4: Very short submissions (< 50 words)
A haiku, a tweet-length micro-story, or a one-paragraph flash fiction piece. The statistical signal can't compute meaningful variance from 3 sentences, so it returns 0.5 ("no information"). The entire classification depends on the LLM signal alone.

**What the system does:** With statistical at 0.5 and LLM as the only real signal, the combined score is `0.5 * 0.4 + llm * 0.6`. Even if the LLM is confident (0.9), the combined score is only 0.74 — just below the AI threshold. The system will say "uncertain" for most short texts regardless of content.

**Design rationale:** This is intentional. We'd rather say "we don't know" about a haiku than make a confident call based on one signal analyzing 17 syllables.

---

## Architecture

### System Flow Diagrams

**Submission flow** — the path a piece of text takes from POST to label:

```
POST /submit { content, creator_id }
        │
        ▼
   ┌─────────────┐
   │ Rate Limiter │──── 429 Too Many Requests
   └──────┬──────┘
          │ content (string), creator_id (string)
          ▼
   ┌──────────────────────────────┐
   │ Signal 1: Statistical        │
   │ - burstiness (CV of sent len)│
   │ - filler phrases (AI phrases) │
   │ - structural (start diversity)│
   └──────┬───────────────────────┘
          │ statistical_score (float 0.0–1.0)
          ▼
   ┌──────────────────────────────┐
   │ Signal 2: LLM (Groq/LLaMA)  │
   │ - semantic/stylistic analysis│
   │ - returns score + rationale  │
   └──────┬───────────────────────┘
          │ llm_score (float 0.0–1.0), rationale (string)
          ▼
   ┌──────────────────────────────┐
   │ Confidence Scoring            │
   │ - weighted avg (0.4 / 0.6)   │
   │ - disagreement adjustment    │
   └──────┬───────────────────────┘
          │ combined_score (float 0.0–1.0)
          ▼
   ┌──────────────────────────────┐
   │ Transparency Label Generator │
   │ score → category → text     │
   │ ≥0.75: "AI-Assisted Content" │
   │ ≤0.25: "Original Work"       │
   │ else:  "Under Review"        │
   └──────┬───────────────────────┘
          │ label_text (string), category (string)
          ▼
   ┌──────────────────────────────┐
   │ Audit Log                     │
   │ event_type: "classification" │
   │ stores: content_hash, both   │
   │ scores, combined, label      │
   └──────┬───────────────────────┘
          │
          ▼
   JSON Response {
     content_id, attribution,
     confidence_score, label, signals
   }
```

**Appeal flow** — the path from appeal to status update:

```
POST /appeal { content_id, creator_reasoning }
        │
        ▼
   ┌─────────────────────────────┐
   │ Validation                   │
   │ - submission exists?         │
   │ - not already under review?  │
   └──────┬──────────────────────┘
          │
          ▼
   ┌─────────────────────────────┐
   │ Status Update                │
   │ submission.status →          │
   │   "under_review"            │
   └──────┬──────────────────────┘
          │
          ▼
   ┌─────────────────────────────┐
   │ Audit Log                    │
   │ event_type: "appeal_submitted│"
   │ stores: appeal_id,           │
   │ appeal_reasoning,            │
   │ original_attribution,       │
   │ original_confidence,         │
   │ status: "under_review"       │
   └──────┬──────────────────────┘
          │
          ▼
   JSON Response {
     appeal_id, content_id,
     status: "under_review", message
   }
```

**Architecture narrative:** Content enters through `POST /submit`, passes through a rate limiter (10/min, 50/hour), then runs sequentially through two independent detection signals — a local statistical analyzer (burstiness, filler phrases, structural patterns) and a remote LLM classifier (Groq/LLaMA). Their scores are combined with weighted averaging (0.4/0.6) and a disagreement adjustment that biases toward uncertainty when signals conflict. The combined score maps to one of three transparency labels designed for non-technical readers. Every decision is captured in a structured audit log with both individual signal scores and the combined result. If a creator contests the result, `POST /appeal` validates the submission exists and isn't already under review, changes the status to "under_review", logs the appeal alongside the original decision, and makes both visible to a reviewer through `GET /status/{content_id}`.

---

## API Contract

### `POST /submit`

**Request body:**
```json
{
  "text": "string (20–50,000 chars, required)",
  "creator_id": "string (optional, defaults to 'anonymous')"
}
```

**Success response (200):**
```json
{
  "content_id": "uuid",
  "attribution": "ai_generated | human_written | uncertain",
  "confidence_score": 0.0–1.0,
  "label": "Full transparency label text",
  "signals": {
    "statistical": 0.0–1.0,
    "llm": 0.0–1.0
  }
}
```

**Error responses:** 400 (missing/invalid content), 429 (rate limited)

### `POST /appeal`

**Request body:**
```json
{
  "content_id": "uuid (required, from a prior /submit response)",
  "creator_reasoning": "string (required, free text explanation)"
}
```

**Success response (200):**
```json
{
  "appeal_id": "uuid",
  "submission_id": "uuid",
  "status": "under_review",
  "message": "Your appeal has been received..."
}
```

**Error responses:** 404 (submission not found), 403 (creator mismatch), 409 (already under review), 400 (missing reason)

### `GET /status/{submission_id}`

**Response (200):** Submission details including attribution, confidence, status, label, and array of related appeals.

### `GET /log`

**Response (200):** `{ "entries": [...], "total": count }` — all audit log entries with timestamps, event types, scores, and decisions.

---

## AI Tool Plan

### Milestone 3: Submission Endpoint + First Signal

**Spec sections to provide:** Detection Signals § Signal 1, Architecture § Submission Flow diagram, API Contract § POST /submit

**What to ask the AI tool to generate:**
- Flask app skeleton with imports, app factory, rate limiter setup
- `analyze_statistical(text)` function implementing all three sub-signals with the exact formulas from the spec (CV for burstiness, TTR for vocab, bigram rate for repetition)
- The `POST /submit` route handler with input validation (20–50,000 chars)
- In-memory `submissions` dict and `audit_log` list

**How to verify before proceeding:**
- Call `analyze_statistical()` directly with three test inputs:
  - A human memoir paragraph (expect score < 0.3 — high burstiness, rich vocab)
  - A formulaic AI-style paragraph with uniform sentences (expect score > 0.5)
  - A 5-word string (expect score = 0.5 with "too short" fallback)
- Confirm sub-signal scores make directional sense (burstiness lower for varied text, repetition higher for repetitive text)
- Start the Flask app, POST to `/submit`, confirm the response shape matches the API contract

### Milestone 4: Second Signal + Confidence Scoring

**Spec sections to provide:** Detection Signals § Signal 2, Uncertainty Representation (full section including worked examples), Architecture § Submission Flow diagram

**What to ask the AI tool to generate:**
- `analyze_llm(text)` function that calls Groq API with the detection prompt, parses JSON response, handles failures gracefully (return 0.5 on error)
- `compute_confidence(stat_score, llm_score)` function implementing weighted average + disagreement adjustment with the exact formula from the spec
- Wire both signals into the existing `/submit` route

**How to verify before proceeding:**
- Test `analyze_llm()` with the same three texts from M3. Check that the LLM score moves in the expected direction.
- Test `compute_confidence()` with the worked examples from section 2:
  - (0.15, 0.20) → ~0.18 (no adjustment, signals agree)
  - (0.20, 0.80) → ~0.49 (strong disagreement pulls to center)
  - (0.80, 0.90) → ~0.86 (no adjustment, signals agree)
- Submit the human memoir via `/submit` — confirm confidence is below 0.25 and attribution is `human_written`
- Submit the AI-style text — confirm it lands in `uncertain` or `ai_generated`
- Check that both signal scores appear in the response

### Milestone 5: Transparency Labels + Appeals + Production Layer

**Spec sections to provide:** Transparency Label Design (all three variants with exact text), Appeals Workflow (full section), API Contract § POST /appeal and GET /status, Architecture § Appeal Flow diagram

**What to ask the AI tool to generate:**
- `generate_label(score)` function returning the exact label text from the spec for each of the three variants, with dynamic confidence percentages
- `POST /appeal/{submission_id}` route with validation (submission exists, creator matches, not already under review)
- `GET /status/{submission_id}` route showing submission details and related appeals
- `GET /log` route returning the audit log
- Rate limiting decorators on submission (10/min, 50/hour) and appeal (5/hour) endpoints

**How to verify before proceeding:**
- Test all three label variants are reachable:
  - Submit text that scores ≥ 0.75 → confirm "AI-Assisted Content" label with percentage
  - Submit text that scores ≤ 0.25 → confirm "Original Work" label with percentage
  - Submit text that scores 0.26–0.74 → confirm "Under Review" label with no percentage
- Test appeal flow end-to-end:
  - Submit content, note the `submission_id`
  - POST appeal with matching `creator_id` → confirm status is `under_review`
  - POST appeal with wrong `creator_id` → confirm 403
  - POST duplicate appeal → confirm 409
  - GET `/status/{id}` → confirm appeal appears in the response with reason and original scores
- Check audit log (`GET /log`) contains at least 3 entries covering classification and appeal events
- Verify rate limiting: submit 11 requests in quick succession → confirm 429 on the 11th
