#!/usr/bin/env python3
"""Generate DALL·E 3 / Flux wild-test images for the Veil detector.

Photorealistic, diverse prompts (the wild set must look like images a user might
try to pass off as real). Resumable: skips indices already on disk. Reads keys
from detector-trainer/.env.

Usage:
    python3 generate_wild.py --provider dalle3 --count 200 --out wild_data
    python3 generate_wild.py --provider flux   --count 400 --out wild_data
    python3 generate_wild.py --provider dalle3 --count 1 --out wild_data  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --- prompt bank ------------------------------------------------------------
SUBJECTS = [
    "an elderly fisherman mending nets on a harbor dock",
    "a young woman reading in a sunlit cafe",
    "a busy street market in a coastal town",
    "a golden retriever running through autumn leaves",
    "a snow-capped mountain range at sunrise",
    "a plate of homemade pasta on a wooden table",
    "a vintage bicycle leaning against a brick wall",
    "a toddler laughing at a birthday party",
    "a rainy city intersection at night with neon reflections",
    "a farmer harvesting wheat in a golden field",
    "a cozy living room with a fireplace and bookshelves",
    "a red fox in a snowy forest",
    "a chef plating a gourmet dish in a restaurant kitchen",
    "a couple hiking on a rocky coastal trail",
    "a close-up of dew on a spider web in the morning",
    "a construction worker on a steel beam at sunset",
    "a bowl of ramen with steam rising",
    "a classic car parked on a suburban driveway",
    "a group of friends at an outdoor barbecue",
    "a lighthouse on a cliff during a storm",
    "a barista pouring latte art",
    "a herd of elephants at a watering hole",
    "a violinist performing on a city sidewalk",
    "a stack of pancakes with berries and syrup",
    "a mountain lake reflecting pine trees",
    "a mechanic repairing a motorcycle in a garage",
    "a bride and groom under a floral arch",
    "a tabby cat sleeping on a windowsill",
    "a fruit vendor arranging produce at a stall",
    "a surfer riding a wave at dawn",
    "a librarian shelving books in a quiet library",
    "a field of sunflowers under a blue sky",
    "a potter shaping clay on a wheel",
    "a train arriving at a rural station",
    "a bowl of fresh salad with grilled chicken",
    "a hot air balloon over a valley at sunrise",
    "a grandmother baking bread in a farmhouse kitchen",
    "a soccer player kicking a ball on a muddy pitch",
    "a quiet alley in an old European city",
    "a hummingbird feeding on a red flower",
]
STYLES = [
    "professional DSLR photograph, natural lighting, high detail",
    "candid smartphone photo, slightly imperfect framing",
    "35mm film photograph, soft grain, warm tones",
    "documentary photography, realistic, shallow depth of field",
]


def build_prompts(count: int) -> list[str]:
    prompts = []
    i = 0
    while len(prompts) < count:
        subj = SUBJECTS[i % len(SUBJECTS)]
        style = STYLES[(i // len(SUBJECTS)) % len(STYLES)]
        prompts.append(f"A realistic photo of {subj}. {style}.")
        i += 1
    return prompts[:count]


def _load_env():
    env = {}
    p = Path(__file__).parent / ".env"
    for line in p.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _download(url: str, dest: Path):
    import urllib.request
    urllib.request.urlretrieve(url, dest)


def gen_dalle3(prompts, out_dir: Path, env):
    """OpenAI image generation. dall-e-3 was retired; use gpt-image-1 (its
    successor). Returns base64, so we decode rather than download a URL."""
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"])
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for idx, prompt in enumerate(prompts):
        dest = out_dir / f"dalle3_{idx:04d}.png"
        if dest.exists():
            done += 1
            continue
        for attempt in range(5):
            try:
                resp = client.images.generate(
                    model="gpt-image-1", prompt=prompt,
                    size="1024x1024", quality="medium", n=1,
                )
                d = resp.data[0]
                if getattr(d, "b64_json", None):
                    dest.write_bytes(base64.b64decode(d.b64_json))
                else:
                    _download(d.url, dest)
                done += 1
                print(f"[dalle3/gpt-image-1] {done}/{len(prompts)} -> {dest.name}", flush=True)
                break
            except Exception as exc:  # rate limits / transient
                wait = min(60, 5 * (attempt + 1))
                print(f"[dalle3] idx {idx} attempt {attempt+1} failed: {exc} — wait {wait}s", flush=True)
                time.sleep(wait)
        time.sleep(1)  # gentle pacing
    return done


def gen_flux(prompts, out_dir: Path, env):
    import replicate
    client = replicate.Client(api_token=env["REPLICATE_API_TOKEN"])
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for idx, prompt in enumerate(prompts):
        dest = out_dir / f"flux_{idx:04d}.jpg"
        if dest.exists():
            done += 1
            continue
        for attempt in range(5):
            try:
                out = client.run(
                    "black-forest-labs/flux-schnell",
                    input={"prompt": prompt, "num_outputs": 1,
                           "aspect_ratio": "1:1", "output_format": "jpg"},
                )
                item = out[0] if isinstance(out, list) else out
                # replicate returns FileOutput (has .read()/.url) or a URL string
                if hasattr(item, "read"):
                    dest.write_bytes(item.read())
                else:
                    _download(str(item), dest)
                done += 1
                print(f"[flux] {done}/{len(prompts)} -> {dest.name}", flush=True)
                break
            except Exception as exc:
                wait = min(30, 3 * (attempt + 1))
                print(f"[flux] idx {idx} attempt {attempt+1} failed: {exc} — wait {wait}s", flush=True)
                time.sleep(wait)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["dalle3", "flux"], required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", default="wild_data", help="base dir; images go to <out>/<provider>/")
    args = ap.parse_args()

    env = _load_env()
    prompts = build_prompts(args.count)
    out_dir = Path(args.out) / args.provider
    print(f"generating {args.count} {args.provider} images -> {out_dir}", flush=True)

    if args.provider == "dalle3":
        n = gen_dalle3(prompts, out_dir, env)
    else:
        n = gen_flux(prompts, out_dir, env)
    print(f"DONE: {n}/{args.count} {args.provider} images in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
