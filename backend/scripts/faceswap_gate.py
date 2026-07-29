"""Acceptance gate for the face-swap signal (spec 2026-07-29 §7).

Scores the reference set with a candidate model and asserts: the Curry composite
is flagged while the three genuine faces are NOT. Run per candidate model id.
"""
import argparse, os, asyncio
from types import SimpleNamespace
from app.signals import faceswap as fs

REF = os.path.expanduser("~/Downloads/photos")
POSITIVE = {"Untitled design.jpg"}                     # composite -> must flag
NEGATIVE = {"sarthakhans_photo.jpg",                   # genuine faces -> must NOT flag
            "WhatsApp Image 2026-06-14 at 12.05.14.jpeg",
            "55195110328_b139fc3758_b.jpg"}


async def main(model_id, threshold):
    fs.get_settings = lambda: SimpleNamespace(
        faceswap_enabled=True, faceswap_model_id=model_id, faceswap_threshold=threshold)
    fs._MODEL = None; fs._DETECTOR = None
    sig = fs.FaceSwapSignal()
    rows = {}
    for name in POSITIVE | NEGATIVE:
        fp = os.path.join(REF, name)
        r = await sig.analyze(fs.ImageInput(data=open(fp, "rb").read(),
                                            filename=fp, content_type=None))
        rows[name] = r.manipulation_score
        print(f"  {r.manipulation_score if r.manipulation_score is not None else 'n/a':>6}  {name}")
    ok_pos = all((rows[n] or 0) >= threshold for n in POSITIVE)
    ok_neg = all((rows[n] or 0) < threshold for n in NEGATIVE)
    print(f"PASS={ok_pos and ok_neg}  (positives flagged={ok_pos}, genuine-faces-safe={ok_neg})")
    return ok_pos and ok_neg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dima806/deepfake_vs_real_image_detection")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    asyncio.run(main(args.model, args.threshold))
