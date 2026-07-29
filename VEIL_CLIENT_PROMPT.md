# Veil — Client Requirements Prompt

You are helping me build **Veil**, an AI media-authenticity assistant. Treat the following as the product vision and client requirements.

## Product Goal

Veil should help ordinary people judge whether an uploaded image is likely authentic, AI-generated, manipulated, or too uncertain to classify. It should combine multiple forms of evidence, explain the result in plain language, and help the user decide what to do next.

Veil is a decision-support tool, not an all-knowing “fake detector.” It must never pretend that a probability is proof.

## Core User Experience

When a user uploads an image, Veil should:

1. Analyze it with the available authenticity detectors, forensic checks, provenance checks, metadata checks, and visual-language model.
2. Evaluate the reliability and relevance of every result before combining them.
3. Return one clear overall assessment:
   - **Likely Authentic**
   - **Low Risk**
   - **Inconclusive**
   - **Medium Risk**
   - **High Risk**
4. Show a calibrated confidence value that reflects actual certainty, not merely an average of conflicting scores.
5. Explain which image-specific evidence affected the result.
6. Clearly disclose missing, failed, unavailable, or contradictory checks.
7. Give practical next steps appropriate to the image and its context.

The experience should feel fast, polished, calm, and trustworthy. The language should be understandable to a nontechnical user.

## Required Detection Behavior

The system must not allow one unreliable model to dominate the final result. A local detector returning 100% AI while strong independent evidence says the image is real should trigger disagreement handling, not an automatic “High Risk” verdict.

The final assessment should account for:

- Model validation quality and known limitations
- Whether the image resembles the model’s training distribution
- Calibration on real phone-camera photos, screenshots, compressed images, edited images, and modern AI-generated images
- Agreement or disagreement among independent checks
- Whether a signal is direct evidence, weak supporting evidence, or merely contextual
- Whether a provider failed or returned no result

An unavailable check must contribute no positive or negative evidence. A failed provider must never silently become a suspicious signal.

Strong disagreement should normally produce **Inconclusive**, alongside a concise explanation of what conflicted. The UI must not represent the midpoint between opposing scores as “50% fake.”

If the system lacks enough reliable evidence, it should say so. Honest uncertainty is better than a confident wrong answer.

## Visual-Language Model Requirements

The VLM should inspect the actual image independently. It must not be told the detector’s predicted label or score before completing its visual inspection, because that could bias its explanation.

The VLM should:

- Describe only observable, image-specific details
- Look for inconsistent text, geometry, reflections, lighting, shadows, anatomy, repeated patterns, object boundaries, and other plausible generation or manipulation artifacts
- Distinguish real evidence from weak visual suspicion
- Avoid generic boilerplate that could apply to any image
- Avoid inventing artifacts that are not visible
- State when no reliable visual evidence is present
- Never claim that appearance alone verifies the source, sender, date, or surrounding story

After the independent inspection, Veil may compare the VLM findings with detector and forensic results. The VLM’s role is to explain visual evidence, not to rationalize a detector score.

## Explanations and Safety Guidance

Every report should answer:

- **Why Veil rated this:** the strongest image-specific evidence and important conflicts
- **Meaning:** what the result does and does not establish
- **Check:** details the user can inspect or independently verify
- **Next:** practical actions the user can take

For high-stakes situations involving money, identity, dating, news, login codes, crypto, gift cards, threats, or urgent requests, Veil should recommend independent verification through another channel. This guidance should be proportional to the evidence and must not make every harmless image sound dangerous.

Technical details may be available under an expandable section, but the primary report should prioritize clarity over raw model output.

## Quality Standard

A normal, unedited iPhone photo of a store shelf must not be labeled “likely false” solely because one internal model outputs 100% AI. That case should be part of the acceptance test suite.

Before Veil presents a detector probability as meaningful, the detector should be evaluated on representative datasets that include:

- Real iPhone and Android camera images
- Social-media recompression
- Screenshots
- Crops and resizes
- JPEG recompression
- Filters and ordinary photo editing
- Multiple current AI-image generators
- Images from sources and generators excluded from training

Evaluation should include false-positive rate, false-negative rate, calibration, per-source performance, robustness under common transformations, and performance on held-out sources. Accuracy alone is not sufficient.

## Product Principles

1. **Evidence before confidence.**
2. **Uncertainty is a valid result.**
3. **Never hide disagreement between checks.**
4. **Do not confuse metadata absence with AI generation.**
5. **Do not confuse editing with full AI generation.**
6. **Do not let explanations merely echo a model score.**
7. **Prefer a cautious, useful answer over a dramatic verdict.**
8. **Make technical evidence understandable without oversimplifying it into a false claim.**
9. **Protect users in high-stakes situations without creating unnecessary fear.**
10. **Validate improvements against real examples and regression tests.**

## Definition of Done

Veil is working ideally when:

- Real-world phone photos are not routinely flagged as AI.
- AI-generated and meaningfully manipulated images are detected with useful, measured confidence.
- Conflicting evidence produces an honest inconclusive result.
- Confidence scores are calibrated and have a documented interpretation.
- VLM explanations are grounded in visible details.
- Provider failures and unavailable checks are transparent.
- Reports are understandable, actionable, and not alarmist.
- Every important scoring behavior is covered by repeatable tests.

Use these requirements as the source of truth when making product, model, UX, or architecture decisions. If a requested implementation would create misleading certainty, hidden disagreement, or unsupported claims, flag the conflict and recommend a safer design.
