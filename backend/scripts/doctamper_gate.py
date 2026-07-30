"""Acceptance gate for the document-tamper signal (spec 2026-07-29 §7).

Scores a reference set and reports whether a threshold exists that flags the
edited document while keeping genuine-but-processed documents below it. If not,
the posture is indicator-only (set doctamper_threshold > 1.0).

Reference layout (assemble under ~/Downloads/doc_ref/):
  edited/    -> at least one genuine document with a Photoshopped field (POSITIVE)
  genuine/   -> the same/other genuine documents, cleanly scanned            (NEGATIVE)
  processed/ -> genuine documents screenshotted and/or JPEG-recompressed     (NEGATIVE)
"""
import os, glob, asyncio
from types import SimpleNamespace
from app.signals import doctamper as dt

REF = os.path.expanduser("~/Downloads/doc_ref")


async def score(path, settings):
    dt.get_settings = lambda: settings
    dt._DOC_GATE = None
    sig = dt.DocTamperSignal()
    r = await sig.analyze(dt.ImageInput(data=open(path, "rb").read(), filename=path, content_type=None))
    return r.manipulation_score


async def main():
    settings = SimpleNamespace(doctamper_enabled=True, doctamper_threshold=0.0,
                               doctamper_doc_gate_threshold=0.55, doctamper_backbone="ViT-L-14")
    groups = {g: sorted(glob.glob(os.path.join(REF, g, "*"))) for g in ("edited", "genuine", "processed")}
    scores = {}
    for g, paths in groups.items():
        scores[g] = []
        for p in paths:
            s = await score(p, settings)
            scores[g].append((os.path.basename(p), s))
            print(f"  {g:10} {s if s is not None else 'n/a':>6}  {os.path.basename(p)}")
    pos = [s for _, s in scores["edited"] if s is not None]
    neg = [s for _, s in scores["genuine"] + scores["processed"] if s is not None]
    if pos and neg:
        margin = min(pos) - max(neg)
        print(f"min(edited)={min(pos):.3f}  max(genuine/processed)={max(neg):.3f}  margin={margin:.3f}")
        print("POSTURE:", "ELEVATE (set threshold between them)" if margin > 0 else "INDICATOR-ONLY (no separating threshold)")


if __name__ == "__main__":
    asyncio.run(main())
