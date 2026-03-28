import base64
import io
import json
import mimetypes
import os
import random
import time
from pathlib import Path

import httpx
from PIL import Image
import yaml
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"
IMAGES_DIR = BACKEND_ROOT / "extracted" / "images"
CAPTIONS_OUT = BACKEND_ROOT / "extracted" / "image_captions.json"


def load_settings() -> dict:
    with open(SETTINGS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_existing_captions() -> dict[str, str]:
    """Load previously generated captions and normalize keys to image filenames."""
    if not CAPTIONS_OUT.exists():
        return {}

    with open(CAPTIONS_OUT, encoding="utf-8") as handle:
        raw = json.load(handle)

    normalized: dict[str, str] = {}
    for key, value in raw.items():
        normalized[Path(key).name] = str(value or "")
    return normalized


def save_captions(captions: dict[str, str]) -> None:
    CAPTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_OUT, "w", encoding="utf-8") as handle:
        json.dump(captions, handle, indent=2, ensure_ascii=False)


def _compute_backoff_delay(
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    strategy: str,
    use_jitter: bool,
) -> float:
    if strategy == "linear":
        delay = base_delay_s * (attempt + 1)
    else:
        delay = base_delay_s * (2**attempt)

    delay = min(delay, max_delay_s)
    if use_jitter:
        delay *= random.uniform(0.75, 1.25)
    return max(0.0, delay)


def _compute_rate_limit_delay(
    attempt: int,
    base_delay_s: float,
    step_delay_s: float,
    max_delay_s: float,
) -> float:
    delay = base_delay_s + (attempt * step_delay_s)
    return max(0.0, min(delay, max_delay_s))


def _is_rate_limited_exception(exc: Exception) -> bool:
    message = str(exc).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _snap28(dim: int) -> int:
    """Round a pixel dimension to the nearest multiple of 28 (min 28)."""
    return max(28, round(dim / 28) * 28)


def encode_image(image_path: Path, max_px: int = 0, snap_to_28: bool = False) -> str:
    """Base64-encode an image.

    If snap_to_28=True, dimensions are rounded to 28-px multiples for Qwen2.5-VL.
    """
    img = Image.open(image_path).convert("RGB")
    if max_px > 0 and max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    w, h = img.size
    if snap_to_28:
        tw, th = _snap28(w), _snap28(h)
        if (tw, th) != (w, h):
            img = img.resize((tw, th), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ensure_model_available(client: httpx.Client, base_url: str, model_name: str) -> None:
    response = client.get(f"{base_url}/api/tags")
    response.raise_for_status()
    available_models = {model.get("name", "") for model in response.json().get("models", [])}
    if model_name not in available_models:
        raise RuntimeError(
            f"Ollama model {model_name} is not available. Pull it first with: ollama pull {model_name}"
        )


def warmup_model(client: httpx.Client, base_url: str, model_name: str) -> None:
    """Prime the full vision pipeline before first real image request."""
    # Use 28x28 so the warmup image aligns with Qwen2.5-VL's patch grid.
    buf = io.BytesIO()
    Image.new("RGB", (28, 28), color=(255, 255, 255)).save(buf, format="PNG")
    tiny_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    try:
        response = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Reply with: ready",
                "images": [tiny_b64],
                "stream": False,
                "keep_alive": "15m",
                "options": {"num_predict": 2},
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        detail = f" body={body[:300]}" if body else ""
        print(f"WARNING: warmup failed ({exc}); proceeding anyway.{detail}")
    except Exception as exc:
        print(f"WARNING: warmup failed ({exc}); proceeding anyway.")


def caption_image_ollama(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    max_retries: int,
    retry_delay_s: int,
    keep_alive: str,
    max_image_px: int = 0,
    retry_strategy: str = "exponential",
    retry_max_delay_s: int = 30,
    retry_use_jitter: bool = True,
) -> str:
    """Generate a caption for a single image with retry/backoff handling."""
    encoded_image = encode_image(image_path, max_px=max_image_px, snap_to_28=True)

    for attempt in range(max_retries):
        try:
            response = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [encoded_image],
                    "stream": False,
                    "keep_alive": keep_alive,
                },
            )
            response.raise_for_status()

            payload = response.json()
            if payload.get("error"):
                raise ValueError(payload["error"])

            caption = str(payload.get("response", "") or "").strip()
            if caption:
                return " ".join(caption.split())

            raise ValueError("Ollama returned an empty caption")
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            print(f"  Attempt {attempt + 1}/{max_retries} transient failure for {image_path.name}: {exc}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            print(f"  Attempt {attempt + 1}/{max_retries} HTTP {status} for {image_path.name}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)
        except Exception as exc:
            print(f"  Attempt {attempt + 1}/{max_retries} failed for {image_path.name}: {exc}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)

    print(f"  WARNING: failed to caption {image_path.name}; leaving it uncached for retry on next run")
    return ""


def caption_image_gemini(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    max_retries: int,
    retry_delay_s: int,
    max_image_px: int = 0,
    retry_strategy: str = "exponential",
    retry_max_delay_s: int = 30,
    retry_use_jitter: bool = True,
    rate_limit_delay_s: int = 60,
    rate_limit_step_delay_s: int = 15,
    rate_limit_max_delay_s: int = 120,
) -> str:
    """Generate a caption for a single image using Gemini with retry/backoff."""
    encoded_image = encode_image(image_path, max_px=max_image_px, snap_to_28=False)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    endpoint = f"{api_base.rstrip('/')}/models/{model_name}:generateContent"

    for attempt in range(max_retries):
        try:
            response = client.post(
                endpoint,
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": encoded_image,
                                    }
                                },
                            ]
                        }
                    ],
                    "generationConfig": {"temperature": 0.1},
                },
            )
            response.raise_for_status()

            payload = response.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                raise ValueError(f"Gemini returned no candidates: {payload}")

            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            text_fragments = [str(part.get("text", "")).strip() for part in parts if isinstance(part, dict)]
            caption = " ".join(fragment for fragment in text_fragments if fragment)
            if caption:
                return " ".join(caption.split())

            raise ValueError("Gemini returned an empty caption")
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            print(f"  Attempt {attempt + 1}/{max_retries} transient failure for {image_path.name}: {exc}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                print(f"  Attempt {attempt + 1}/{max_retries} rate limited for {image_path.name}: HTTP 429")
                if attempt < max_retries - 1:
                    delay_s = _compute_rate_limit_delay(
                        attempt=attempt,
                        base_delay_s=float(rate_limit_delay_s),
                        step_delay_s=float(rate_limit_step_delay_s),
                        max_delay_s=float(rate_limit_max_delay_s),
                    )
                    print(f"    backing off for {delay_s:.1f}s")
                    time.sleep(delay_s)
                continue

            print(f"  Attempt {attempt + 1}/{max_retries} HTTP {status} for {image_path.name}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)
        except Exception as exc:
            if _is_rate_limited_exception(exc):
                print(f"  Attempt {attempt + 1}/{max_retries} rate limited for {image_path.name}: {exc}")
                if attempt < max_retries - 1:
                    delay_s = _compute_rate_limit_delay(
                        attempt=attempt,
                        base_delay_s=float(rate_limit_delay_s),
                        step_delay_s=float(rate_limit_step_delay_s),
                        max_delay_s=float(rate_limit_max_delay_s),
                    )
                    print(f"    backing off for {delay_s:.1f}s")
                    time.sleep(delay_s)
                continue

            print(f"  Attempt {attempt + 1}/{max_retries} failed for {image_path.name}: {exc}")
            if attempt < max_retries - 1:
                delay_s = _compute_backoff_delay(
                    attempt=attempt,
                    base_delay_s=float(retry_delay_s),
                    max_delay_s=float(retry_max_delay_s),
                    strategy=retry_strategy,
                    use_jitter=retry_use_jitter,
                )
                print(f"    retrying in {delay_s:.1f}s")
                time.sleep(delay_s)

    print(f"  WARNING: failed to caption {image_path.name}; leaving it uncached for retry on next run")
    return ""


def run() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    cfg = load_settings()
    model_cfg = cfg.get("models", {})
    ollama_cfg = cfg.get("ollama", {})
    caption_cfg = cfg.get("captioning", {})

    base_url = str(os.getenv("OLLAMA_BASE_URL", ollama_cfg.get("base_url", "http://localhost:11434"))).rstrip("/")
    provider = str(caption_cfg.get("provider", "ollama")).strip().lower()
    caption_model = str(caption_cfg.get("model") or model_cfg.get("captioning", "qwen2.5vl:7b")).strip()
    prompt = str(
        caption_cfg.get(
            "prompt",
            "Describe this technical diagram precisely. Include all visible labels and relationships.",
        )
    ).strip()
    max_retries = int(caption_cfg.get("max_retries", 3))
    retry_delay_s = int(caption_cfg.get("retry_delay_s", 2))
    retry_strategy = str(caption_cfg.get("retry_strategy", "exponential")).strip().lower()
    retry_max_delay_s = int(caption_cfg.get("retry_max_delay_s", 30))
    retry_use_jitter = bool(caption_cfg.get("retry_use_jitter", True))
    rate_limit_delay_s = int(caption_cfg.get("rate_limit_delay_s", 60))
    rate_limit_step_delay_s = int(caption_cfg.get("rate_limit_step_delay_s", 15))
    rate_limit_max_delay_s = int(caption_cfg.get("rate_limit_max_delay_s", 120))
    inter_image_delay_s = float(caption_cfg.get("inter_image_delay_s", 5))
    request_timeout_s = int(caption_cfg.get("request_timeout_s", ollama_cfg.get("timeout", 60)))
    connect_timeout_s = int(caption_cfg.get("connect_timeout_s", 20))
    keep_alive = str(caption_cfg.get("keep_alive", "15m"))
    max_image_px = int(caption_cfg.get("max_image_px", 1024))

    gemini_cfg = cfg.get("gemini", {})
    gemini_api_base = str(gemini_cfg.get("api_base", "https://generativelanguage.googleapis.com/v1beta")).rstrip("/")
    gemini_api_key_env = str(gemini_cfg.get("api_key_env", "GEMINI_API_KEY"))
    gemini_api_key = str(os.getenv(gemini_api_key_env, "")).strip()

    if not IMAGES_DIR.exists():
        print(f"Images directory not found: {IMAGES_DIR}")
        return

    image_files = sorted(
        [
            p
            for p in IMAGES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )

    if not image_files:
        print(f"No image files found in {IMAGES_DIR}")
        return

    captions = load_existing_captions()
    print(f"Found {len(image_files)} images. {len(captions)} already captioned.")
    run_started = time.perf_counter()

    timeout = httpx.Timeout(connect=connect_timeout_s, read=request_timeout_s, write=request_timeout_s, pool=connect_timeout_s)
    with httpx.Client(timeout=timeout) as client:
        if provider == "ollama":
            ensure_model_available(client, base_url, caption_model)
            warmup_model(client, base_url, caption_model)
        elif provider == "gemini":
            if not gemini_api_key:
                raise RuntimeError(
                    f"Gemini API key is missing. Set environment variable {gemini_api_key_env}."
                )
        else:
            raise ValueError(f"Unsupported captioning provider: {provider}. Use 'ollama' or 'gemini'.")

        for idx, image_path in enumerate(image_files, start=1):
            filename = image_path.name
            existing = str(captions.get(filename, "") or "").strip()
            if existing:
                print(f"  [{idx}/{len(image_files)}] skip (cached): {filename}")
                continue

            item_started = time.perf_counter()
            print(f"  [{idx}/{len(image_files)}] captioning: {filename}")
            caption_result = ""
            try:
                if provider == "ollama":
                    caption_result = caption_image_ollama(
                        client=client,
                        base_url=base_url,
                        model_name=caption_model,
                        prompt=prompt,
                        image_path=image_path,
                        max_retries=max_retries,
                        retry_delay_s=retry_delay_s,
                        keep_alive=keep_alive,
                        max_image_px=max_image_px,
                        retry_strategy=retry_strategy,
                        retry_max_delay_s=retry_max_delay_s,
                        retry_use_jitter=retry_use_jitter,
                    )
                else:
                    caption_result = caption_image_gemini(
                        client=client,
                        api_base=gemini_api_base,
                        api_key=gemini_api_key,
                        model_name=caption_model,
                        prompt=prompt,
                        image_path=image_path,
                        max_retries=max_retries,
                        retry_delay_s=retry_delay_s,
                        max_image_px=max_image_px,
                        retry_strategy=retry_strategy,
                        retry_max_delay_s=retry_max_delay_s,
                        retry_use_jitter=retry_use_jitter,
                        rate_limit_delay_s=rate_limit_delay_s,
                        rate_limit_step_delay_s=rate_limit_step_delay_s,
                        rate_limit_max_delay_s=rate_limit_max_delay_s,
                    )
            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving progress before exit...")
                save_captions(captions)
                return
            item_elapsed = time.perf_counter() - item_started
            if caption_result:
                captions[filename] = caption_result
                save_captions(captions)
                print(f"  [{idx}/{len(image_files)}] done in {item_elapsed:.1f}s: {filename}")
            else:
                print(f"  [{idx}/{len(image_files)}] no caption saved in {item_elapsed:.1f}s: {filename}")
                print(f"    skipping save for {filename}; it will be retried on the next run")

            if inter_image_delay_s > 0 and idx < len(image_files):
                print(f"    sleeping {inter_image_delay_s:.1f}s before next image")
                time.sleep(inter_image_delay_s)

    total_elapsed = time.perf_counter() - run_started
    print(f"Done in {total_elapsed:.1f}s. Saved {len(captions)} captions to {CAPTIONS_OUT}")


if __name__ == "__main__":
    run()
