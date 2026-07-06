# Provenance Guard

A backend API that classifies whether submitted text content is AI-generated or human-written, produces transparency labels for end users, and supports creator appeals.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Run the server:
```bash
python3 app.py
```

The server starts on `http://localhost:5001`.

## How It Works

A piece of text flows through the system like this:

1. Creator submits text via `POST /submit`
2. Rate limiter checks submission frequency (10/min, 50/hour)
3. **Signal 1 (Statistical Analysis)** measures burstiness, filler phrase density, and structural patterns locally
4. **Signal 2 (LLM Classification)** sends text to Groq/LLaMA for semantic and stylistic analysis
5. Confidence scoring combines both signals with weighted averaging and a disagreement adjustment
6. A transparency label is generated based on the confidence score
7. The full decision is recorded in the audit log
8. The response includes attribution, confidence, label, and both signal scores

## Detection Signals

### Why two signals?

Single-signal detection is brittle. A statistical analyzer can be fooled by a human who writes formally; an LLM detector can be fooled by text from a similar model family. By requiring both to agree before making a confident call, we reduce the false positive rate — the most harmful failure mode on a creative writing platform, where accusing a human of using AI undermines trust.

### Signal 1: Statistical Text Analysis

Runs locally with no API call. Measures three text properties:

- **Burstiness (sentence length variance):** Human writers produce sentences of wildly varying lengths — "I ran." followed by a 20-word sentence. AI tends toward uniform 10-15 word sentences. We compute the coefficient of variation (std_dev / mean) of sentence word counts. Score = `clamp(1.0 - cv/0.8, 0, 1)`. Weight: 35%.
- **Filler Phrase Density:** AI text uses characteristic transition/hedge phrases ("it is important to note," "furthermore," "in conclusion," "a wide range of") at much higher rates than human creative writing. We check for 20 known AI filler phrases and compute hits per sentence. Score = `clamp(density / 1.5, 0, 1)`. Weight: 35%.
- **Structural Patterns:** Combines sentence-start diversity (do sentences begin with varied words?) and length uniformity. AI text often starts multiple sentences the same way ("The," "It," "This") and maintains consistent lengths. Weight: 30%.

Each sub-signal produces a score from 0.0 (strongly human) to 1.0 (strongly AI), weighted and averaged into a single statistical score.

**Why these sub-signals?** I initially implemented type-token ratio (vocabulary richness) and bigram repetition rate, but testing showed both returned 0.0 for every input under 200 words — they simply don't differentiate at the text lengths a writing platform sees. I replaced them with filler phrase detection and structural patterns, which produce meaningful scores even on 3-sentence inputs. Burstiness remained because it was the strongest differentiator from the start.

**Blind spots:** Highly edited human text (academic, legal) can look uniform like AI. Short texts (< 50 words) lack enough data — the signal returns 0.5 ("no information"). Non-native English speakers may trigger false positives on structural patterns.

### Signal 2: LLM-Based Classification

Sends the text (up to 3,000 chars) to Groq (LLaMA 3.1 8B) with a calibrated prompt that includes a scoring guide with concrete examples at each level:
- 0.0-0.2: Clearly human (messy, informal, personal voice)
- 0.3-0.4: Probably human (well-written with human quirks)
- 0.5: Truly uncertain
- 0.6-0.7: Probably AI (suspiciously polished, hedging language)
- 0.8-1.0: Clearly AI (formulaic transitions, no personal voice)

The prompt lists specific strong AI indicators ("Furthermore," "Moreover," parallel list structure, absence of contractions) and strong human indicators (slang, sentence fragments, humor, specific personal anecdotes).

**Why this prompt design?** My first prompt was generic ("determine how likely it is to be AI-generated"). The LLM scored everything in the 0.2-0.6 range — it never used the extremes. Adding the explicit scoring guide with examples at each level fixed this: the "clearly AI" test input went from 0.6 to 0.9, and "clearly human" went from 0.3 to 0.0. The calibration guide teaches the model what each part of the scale means.

**Blind spots:** The detector is itself an LLM, so same-family text may be harder to catch. Polished human writing can read as "too perfect." If the Groq API is unreachable, the signal returns 0.5 (safe fallback to uncertainty).

**What I'd change for production:** I'd use a fine-tuned classifier instead of prompt-based detection — prompt engineering is fragile and model-dependent. I'd also add confidence calibration by testing against a labeled dataset and adjusting the score distribution to match empirical accuracy rates.

## Confidence Scoring

The two signals are combined with weighted averaging (statistical: 0.4, LLM: 0.6). The LLM gets higher weight because it captures semantic patterns (hedging, voice, structure) that statistics miss, while the statistical signal grounds the analysis with deterministic, objective measurements.

**Disagreement adjustment:** When the two signals disagree by more than 0.3, the combined score is pulled toward 0.5 (uncertain):
```
disagreement = abs(stat_score - llm_score)
if disagreement > 0.3:
    pull_factor = min(1.0, (disagreement - 0.3) / 0.4)
    weighted = weighted + (0.5 - weighted) * pull_factor * 0.5
```

This means:
- A score of 0.06 represents "both signals strongly agree this is human-written" (94% confidence)
- A score of 0.54 represents "the signals disagree significantly — we genuinely aren't sure"
- A score of 0.84 represents "both signals strongly agree this is AI-generated" (84% confidence)

This design reflects the false-positive asymmetry: falsely labeling a human's work as AI is worse than missing AI content. When in doubt, the system says "uncertain" rather than making an accusation.

**Example 1 — High-confidence result (human-written):**

Input: *"ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after."*

```
Statistical: 0.14  (low burstiness, no filler phrases, varied structure)
LLM:         0.00  (slang, sentence fragments, personal anecdote, humor)
Combined:    0.06  → "human_written" (94% confidence)
```

Both signals strongly agree. The statistical signal sees varied sentence lengths and no AI filler phrases. The LLM sees slang ("honestly?"), capitalization for emphasis ("WAY"), and a specific personal story. No disagreement adjustment needed.

**Example 2 — High-confidence result (AI-generated):**

Input: *"It is important to note that artificial intelligence has become increasingly significant. Furthermore, the integration of AI systems into various industries has created both opportunities and challenges. Additionally, it should be noted that..."*

```
Statistical: 0.75  (uniform sentences, heavy filler phrases, repetitive starts)
LLM:         0.90  (formulaic transitions, no personal voice, template structure)
Combined:    0.84  → "ai_generated" (84% confidence)
```

Both signals strongly agree. The statistical signal catches 6+ filler phrases in 6 sentences and uniform sentence lengths. The LLM sees the classic AI pattern: "Furthermore," "Additionally," "Moreover," hedging language, no personal voice.

**Full test matrix (4 inputs from milestone spec):**

| Input | Statistical | LLM | Combined | Attribution |
|---|---|---|---|---|
| Clearly AI (formulaic transitions, hedging) | 0.51 | 0.90 | 0.72 | uncertain (near AI threshold) |
| Clearly human (informal ramen review) | 0.14 | 0.00 | 0.06 | human_written (94% confidence) |
| Formal human (monetary policy paragraph) | 0.35 | 0.50 | 0.44 | uncertain |
| Edited AI (remote work reflection) | 0.29 | 0.20 | 0.24 | human_written (borderline) |

The scores span the full 0.06–0.84 range. Note that the 3-sentence "clearly AI" text scores 0.72 (uncertain) because the statistical signal only reaches 0.51 — the text is too short for all sub-signals to fire confidently. The longer, more heavily AI-patterned text above scores 0.84 because both signals agree more strongly.

## Transparency Labels

Three label variants are displayed to readers:

### High-Confidence AI (score >= 0.75)
> "This content has been flagged as likely AI-generated. Our analysis detected patterns consistent with AI writing tools. Confidence: 84%. If you believe this is incorrect, you can submit an appeal."

Uses "flagged as likely" rather than "is" to communicate probability, not certainty. Includes the confidence percentage so readers can judge for themselves. Mentions the appeal path so creators see recourse immediately.

### High-Confidence Human (score <= 0.25)
> "This content appears to be human-written. Our analysis found patterns consistent with original human authorship. Confidence: 94%."

Uses "appears to be" to maintain epistemic honesty. Confidence is calculated as `(1 - score) * 100` so the percentage always represents confidence in the stated verdict.

### Uncertain (0.25 < score < 0.75)
> "We couldn't determine with confidence whether this content is AI-generated or human-written. This may reflect mixed authorship, an unusual writing style, or heavy editing. The creator can provide additional context through an appeal."

Deliberately non-accusatory. Lists three legitimate, non-stigmatizing reasons for uncertainty. No percentage shown — displaying "52% AI" would mislead readers into treating noise as signal. Points the creator to the appeal system.

## Appeals Workflow

Creators who believe they've been misclassified can submit an appeal:

1. `POST /appeal` with `content_id` and `creator_reasoning`
2. The system validates the submission exists and isn't already under review
3. The submission status changes to `"under_review"`
4. The appeal is logged in the audit trail alongside the original classification
5. A moderator can review via `GET /status/{content_id}` which shows the original decision, both signal scores, and the creator's reasoning

Duplicate appeals on the same submission are rejected (HTTP 409).

**Example appeal:**
```bash
curl -s -X POST http://localhost:5001/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}' | python3 -m json.tool
```

**Response:**
```json
{
  "appeal_id": "f9d2640d-...",
  "content_id": "aeb3ec8a-...",
  "status": "under_review",
  "message": "Your appeal has been received and the content's classification has been placed under review. A moderator will evaluate your submission with the additional context you provided."
}
```

## Rate Limiting

| Endpoint | Limit | Reasoning |
|----------|-------|-----------|
| `POST /submit` | 10/min, 50/hour | A creator realistically submits a few pieces per session. 10/min prevents automated flooding while allowing batch submissions. 50/hour caps sustained abuse. |
| `POST /appeal` | 5/hour | Appeals require thought — 5/hour is generous for legitimate use while preventing appeal spam. |
| Global | 100/hour | Backstop for all endpoints combined. |

These values assume a writing platform where creators submit finished pieces, not a high-throughput API. An adversary trying to flood the system hits the per-minute limit first, then the hourly cap.

**Rate limit test output (12 requests, 10/min limit):**
```
Request 1: 200
Request 2: 200
Request 3: 200
Request 4: 200
Request 5: 200
Request 6: 200
Request 7: 200
Request 8: 429
Request 9: 429
Request 10: 429
Request 11: 429
Request 12: 429
```
(Requests 8-12 returned 429 because the test ran after prior submissions had already consumed part of the 10/min quota.)

## Audit Log

Every decision is captured in a structured audit log accessible via `GET /log`. Each entry includes:

- `id`: Unique event ID (UUID)
- `timestamp`: ISO 8601 UTC timestamp
- `event_type`: `"classification"` or `"appeal_submitted"`
- `content_id`: Links to the content
- `creator_id`: Who submitted the content
- `content_hash`: SHA-256 hash prefix for content integrity
- For classifications: `statistical_score`, `llm_score`, `combined_score`, `attribution`, `label_text`
- For appeals: `appeal_id`, `appeal_reasoning`, `original_attribution`, `original_confidence`, `status`

### Sample Audit Log (3 entries)

```json
[
  {
    "event_type": "classification",
    "content_id": "92625d06-...",
    "creator_id": "label-test-1",
    "content_hash": "819676fbe89024f8",
    "statistical_score": 0.1409,
    "llm_score": 0.0,
    "combined_score": 0.0564,
    "attribution": "human_written",
    "label_text": "This content appears to be human-written. Our analysis found patterns consistent with original human authorship. Confidence: 94%.",
    "timestamp": "2026-07-06T01:21:09.738208+00:00"
  },
  {
    "event_type": "classification",
    "content_id": "aeb3ec8a-...",
    "creator_id": "label-test-3",
    "content_hash": "4ae9bf261fd13d9c",
    "statistical_score": 0.7475,
    "llm_score": 0.9,
    "combined_score": 0.839,
    "attribution": "ai_generated",
    "label_text": "This content has been flagged as likely AI-generated. Our analysis detected patterns consistent with AI writing tools. Confidence: 84%. If you believe this is incorrect, you can submit an appeal.",
    "timestamp": "2026-07-06T01:21:10.130991+00:00"
  },
  {
    "event_type": "appeal_submitted",
    "content_id": "aeb3ec8a-...",
    "appeal_id": "f9d2640d-...",
    "creator_id": "label-test-3",
    "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
    "original_attribution": "ai_generated",
    "original_confidence": 0.839,
    "status": "under_review",
    "timestamp": "2026-07-06T01:21:22.049945+00:00"
  }
]
```

## Known Limitations

### Non-native English speakers get false positives

A non-native English speaker writing in a formal, careful style — avoiding contractions, using simple vocabulary, constructing uniform sentence lengths — will trigger both the statistical signal (low burstiness, high structural uniformity) and the LLM signal (absence of contractions and personal voice reads as "AI-like"). The system can't distinguish "writes formally because English is their second language" from "writes formally because an AI generated it." This is the most concerning failure mode because it disproportionately affects a specific group of creators.

The appeals system partially mitigates this: the uncertain label directs the creator to explain their writing style. But the system shouldn't need an appeal to get this right — a production version would need to learn from appeal outcomes or incorporate authorship metadata.

### Intentionally repetitive poetry

A poet using repetition as a literary device (think: villanelles, anaphora, refrains) will score high on the filler phrase and structural pattern sub-signals. The system interprets repeated sentence structures as AI uniformity, not deliberate craft. A haiku with 17 syllables hits the short-text fallback (returns 0.5) and can't be meaningfully classified at all.

### The LLM detector is itself an LLM

Signal 2 uses LLaMA 3.1 8B to detect AI text — but text generated by the same model family may share enough "voice" that the detector can't distinguish self from other. This is an inherent limitation of using LLMs to detect LLM output. A production system would need model-diverse detection (multiple LLM families, or non-LLM classifiers trained on labeled data).

### In-memory storage doesn't persist

All submissions, appeals, and audit log entries are stored in Python dicts — they vanish when the server restarts. A production system would use a database. This is a prototype limitation, not an architectural one.

## Spec Reflection

### Where the spec helped

The planning.md section on uncertainty representation — specifically the worked examples table showing what happens when signals agree vs. disagree — directly guided the implementation of the disagreement adjustment formula. When I was coding `compute_confidence()`, I could test it against those six input pairs and immediately see whether the output matched what I'd designed. Without those concrete examples, I would have tuned the formula by feel, which is much harder to verify.

### Where the implementation diverged

The spec originally described Signal 1 as using type-token ratio (vocabulary richness) and bigram repetition rate as sub-signals. When I tested these on real inputs, both returned 0.0 for every text under 200 words — the TTR of a 50-word passage is naturally high (most words are unique), and bigrams almost never repeat in short text. I replaced them with filler phrase detection and structural pattern analysis, which produce meaningful scores even on 3-sentence inputs. The spec was correct about *what* to measure (lexical patterns that differ between AI and human text) but wrong about *how* to measure it at the text lengths this system actually encounters. I updated planning.md to reflect the change after implementing it.

## AI Usage

### Instance 1: Initial application scaffold

I provided the detection signals section and architecture diagram from planning.md and asked Claude to generate the Flask app skeleton with both signal functions, the confidence scoring logic, and all route handlers. The generated code was structurally correct — routes matched the API contract, the signal functions returned the right shape — but the statistical sub-signals (TTR, bigram repetition) produced 0.0 on every test input. I diagnosed this by calling `analyze_statistical()` directly and printing intermediate values, discovered the formulas didn't differentiate at short text lengths, and replaced both sub-signals with filler phrase detection and structural pattern analysis that I designed and tested independently.

### Instance 2: LLM prompt calibration

I asked Claude to write the Groq prompt for Signal 2. The initial prompt was generic: "determine how likely it is to be AI-generated." Testing showed the LLM scored everything between 0.2–0.6 — it never used the extremes. I rewrote the prompt myself with an explicit scoring guide (0.0-0.2: clearly human characteristics, 0.8-1.0: clearly AI characteristics) and lists of concrete indicators (contractions/slang/fragments for human, "Furthermore"/"Moreover"/hedging for AI). This pushed the clearly-AI test input from 0.6 to 0.9 and the clearly-human input from 0.3 to 0.0. The calibration guide was the key insight — without it, the LLM defaults to a narrow, safe range.

### Instance 3: Appeal endpoint field names

The milestone spec used `creator_reasoning` as the appeal field name and `POST /appeal` with `content_id` in the body (not the URL). My initial implementation used `reason` and `POST /appeal/<content_id>`. I caught this by reading the milestone's example curl command carefully and updated the code to accept both `creator_reasoning` and `reason` (for backward compatibility), and moved `content_id` into the request body to match the spec exactly.

## API Reference

### `POST /submit`
Submit content for attribution analysis.

**Request:**
```json
{
  "text": "Your text here (20-50,000 characters)",
  "creator_id": "optional-creator-id"
}
```

**Response:**
```json
{
  "content_id": "uuid",
  "attribution": "ai_generated | human_written | uncertain",
  "confidence_score": 0.0-1.0,
  "label": "Transparency label text",
  "signals": {
    "statistical": 0.0-1.0,
    "llm": 0.0-1.0
  }
}
```

### `POST /appeal`
Appeal a classification decision.

**Request:**
```json
{
  "content_id": "uuid from a prior /submit response",
  "creator_reasoning": "Why you believe the classification is wrong"
}
```

**Response:**
```json
{
  "appeal_id": "uuid",
  "content_id": "uuid",
  "status": "under_review",
  "message": "Your appeal has been received..."
}
```

### `GET /status/{content_id}`
Check a submission's current status and any appeals.

### `GET /log`
View the full audit log.
