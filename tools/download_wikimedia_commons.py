#!/usr/bin/env python3
"""Download a small color-image dataset from Wikimedia Commons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Protocol

from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError
import requests

API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "tennis-lab-dino-ssl-bot/1.1 "
    "(https://github.com/Motoki0705/tennis-lab; Wikimedia dataset builder)"
)
SUPPORTED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
SEARCH_PAGE_LIMIT = 10_000


@dataclass(frozen=True)
class CommonsFile:
    page_id: int
    title: str
    download_url: str
    description_url: str
    mime: str
    width: int
    height: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class ManifestEntry:
    filename: str
    sha256: str
    width: int
    height: int
    color_difference: float
    commons_page_id: int
    commons_title: str
    source_url: str
    description_url: str
    source_mime: str
    license: str
    license_url: str
    artist: str
    credit: str


@dataclass(frozen=True)
class ResumeState:
    entries: list[ManifestEntry]
    page_ids: set[int]
    pixel_hashes: set[str]


class CommonsClient(Protocol):
    def search_files(self, query: str, limit: int) -> Iterable[CommonsFile]: ...

    def download(self, url: str) -> bytes: ...


class AdaptiveRateLimiter:
    def __init__(
        self,
        *,
        initial_delay: float,
        min_delay: float,
        max_delay: float,
    ) -> None:
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_request_at = 0.0

    def wait(self) -> None:
        wait_seconds = self.delay - (time.monotonic() - self._last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.monotonic()

    def record_success(self) -> None:
        self.delay = max(self.min_delay, self.delay * 0.98)

    def record_throttle(self, retry_after: float | None) -> float:
        delay = retry_after if retry_after is not None else max(1.0, self.delay * 2)
        self.delay = min(self.max_delay, max(self.delay, delay))
        return self.delay


class WikimediaCommonsClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 30.0,
        retries: int = 3,
        api_request_delay: float = 0.2,
        image_request_delay: float = 0.5,
        thumbnail_width: int = 512,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A descriptive Wikimedia User-Agent is required")
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.thumbnail_width = thumbnail_width
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip",
            }
        )
        self._api_limiter = AdaptiveRateLimiter(
            initial_delay=api_request_delay,
            min_delay=0.1,
            max_delay=30.0,
        )
        self._image_limiter = AdaptiveRateLimiter(
            initial_delay=image_request_delay,
            min_delay=0.25,
            max_delay=60.0,
        )

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _request(
        self,
        url: str,
        *,
        limiter: AdaptiveRateLimiter,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        for attempt in range(self.retries + 1):
            limiter.wait()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    limiter.record_success()
                    return response.content
                if attempt >= self.retries:
                    response.raise_for_status()
                delay = limiter.record_throttle(self._retry_after_seconds(response))
                print(
                    f"HTTP {response.status_code}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.retries})",
                    flush=True,
                )
            except requests.RequestException:
                if attempt >= self.retries:
                    raise
                delay = min(limiter.max_delay, max(1.0, 2**attempt))
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        return json.loads(
            self._request(
                API_URL,
                limiter=self._api_limiter,
                params=params,
            )
        )

    @staticmethod
    def _search_expressions(query: str) -> tuple[str, ...]:
        normalized = " ".join(query.split())
        expressions = (
            f'"{normalized}"',
            normalized,
            f'"{normalized}s"',
            f"{normalized} stadium",
            f"{normalized} club",
            f"{normalized} tournament",
            f"indoor {normalized}",
            f"outdoor {normalized}",
        )
        return tuple(dict.fromkeys(expressions))

    def search_files(self, query: str, limit: int) -> Iterable[CommonsFile]:
        seen_page_ids: set[int] = set()
        yielded = 0
        for search_expression in self._search_expressions(query):
            continuation: dict[str, Any] = {}
            expression_yielded = 0
            print(f'Searching Wikimedia Commons for: {search_expression}', flush=True)
            while yielded < limit and expression_yielded < SEARCH_PAGE_LIMIT:
                params: dict[str, Any] = {
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "generator": "search",
                    "gsrsearch": search_expression,
                    "gsrnamespace": 6,
                    "gsrlimit": min(50, limit - yielded),
                    "prop": "imageinfo",
                    "iiprop": "url|mime|size|extmetadata",
                    "iiurlwidth": self.thumbnail_width,
                    **continuation,
                }
                result = self._request_json(params)
                for page in result.get("query", {}).get("pages", []):
                    page_id = int(page["pageid"])
                    expression_yielded += 1
                    if page_id in seen_page_ids:
                        continue
                    seen_page_ids.add(page_id)
                    image_info = page.get("imageinfo", [])
                    if not image_info:
                        continue
                    info = image_info[0]
                    metadata = {
                        key: str(value.get("value", ""))
                        for key, value in info.get("extmetadata", {}).items()
                    }
                    download_url = info.get("thumburl") or info.get("url")
                    if not download_url:
                        continue
                    yield CommonsFile(
                        page_id=page_id,
                        title=str(page["title"]),
                        download_url=str(download_url),
                        description_url=str(info.get("descriptionurl", "")),
                        mime=str(info.get("mime", "")),
                        width=int(info.get("width", 0)),
                        height=int(info.get("height", 0)),
                        metadata=metadata,
                    )
                    yielded += 1
                    if yielded >= limit:
                        return
                continuation = result.get("continue", {})
                if not continuation:
                    break

    def download(self, url: str) -> bytes:
        return self._request(url, limiter=self._image_limiter)


def _color_difference(image: Image.Image) -> float:
    sample = image.resize((64, 64))
    red, green, blue = sample.split()
    differences = (
        ImageStat.Stat(ImageChops.difference(red, green)).mean[0],
        ImageStat.Stat(ImageChops.difference(red, blue)).mean[0],
        ImageStat.Stat(ImageChops.difference(green, blue)).mean[0],
    )
    return sum(differences) / len(differences)


def _prepare_image(
    data: bytes,
    *,
    min_dimension: int,
    min_color_difference: float,
    output_max_dimension: int,
) -> tuple[Image.Image, float] | None:
    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None

    if min(image.size) < min_dimension:
        return None
    color_difference = _color_difference(image)
    if color_difference < min_color_difference:
        return None
    image.thumbnail((output_max_dimension, output_max_dimension))
    return image, color_difference


def _metadata_value(file: CommonsFile, key: str) -> str:
    return file.metadata.get(key, "")


def _write_manifest(
    *,
    output_dir: Path,
    query: str,
    entries: list[ManifestEntry],
    complete: bool,
) -> None:
    manifest = {
        "source": "Wikimedia Commons",
        "query": query,
        "complete": complete,
        "num_images": len(entries),
        "images": [asdict(entry) for entry in entries],
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def _load_resume_state(output_dir: Path) -> ResumeState:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Resume manifest does not exist: "{manifest_path}"')

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [ManifestEntry(**item) for item in manifest.get("images", [])]
    page_ids: set[int] = set()
    pixel_hashes: set[str] = set()
    for entry in entries:
        image_path = output_dir / entry.filename
        if not image_path.is_file():
            raise FileNotFoundError(f'Resume image does not exist: "{image_path}"')
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        page_ids.add(entry.commons_page_id)
        pixel_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
    return ResumeState(entries=entries, page_ids=page_ids, pixel_hashes=pixel_hashes)


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_dataset(
    *,
    client: CommonsClient,
    query: str,
    output_dir: Path,
    max_images: int,
    candidate_multiplier: int,
    min_dimension: int,
    min_color_difference: float,
    output_max_dimension: int,
    jpeg_quality: int,
    manifest_interval: int = 25,
    progress_interval: int = 25,
    resume_state: ResumeState | None = None,
) -> list[ManifestEntry]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if resume_state is None:
        entries: list[ManifestEntry] = []
        seen_page_ids: set[int] = set()
        seen_hashes: set[str] = set()
    else:
        entries = list(resume_state.entries)
        seen_page_ids = set(resume_state.page_ids)
        seen_hashes = set(resume_state.pixel_hashes)
        print(f"Resuming from {len(entries)} existing images", flush=True)
    if len(entries) >= max_images:
        _write_manifest(
            output_dir=output_dir,
            query=query,
            entries=entries,
            complete=True,
        )
        return entries
    starting_count = len(entries)
    candidate_limit = max(max_images, max_images * candidate_multiplier)
    candidates_seen = 0
    started_at = time.monotonic()

    for commons_file in client.search_files(query, candidate_limit):
        candidates_seen += 1
        if commons_file.page_id in seen_page_ids:
            continue
        if commons_file.mime not in SUPPORTED_MIME_TYPES:
            continue
        if min(commons_file.width, commons_file.height) < min_dimension:
            continue
        try:
            data = client.download(commons_file.download_url)
        except (requests.RequestException, TimeoutError):
            continue
        prepared = _prepare_image(
            data,
            min_dimension=min_dimension,
            min_color_difference=min_color_difference,
            output_max_dimension=output_max_dimension,
        )
        if prepared is None:
            continue
        image, color_difference = prepared
        pixel_digest = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_digest in seen_hashes:
            continue

        filename = f"{len(entries):06d}.jpg"
        image_path = images_dir / filename
        image.save(image_path, format="JPEG", quality=jpeg_quality)
        file_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        seen_page_ids.add(commons_file.page_id)
        seen_hashes.add(pixel_digest)
        entries.append(
            ManifestEntry(
                filename=f"images/{filename}",
                sha256=file_digest,
                width=image.width,
                height=image.height,
                color_difference=round(color_difference, 4),
                commons_page_id=commons_file.page_id,
                commons_title=commons_file.title,
                source_url=commons_file.download_url,
                description_url=commons_file.description_url,
                source_mime=commons_file.mime,
                license=_metadata_value(commons_file, "LicenseShortName"),
                license_url=_metadata_value(commons_file, "LicenseUrl"),
                artist=_metadata_value(commons_file, "Artist"),
                credit=_metadata_value(commons_file, "Credit"),
            )
        )
        if len(entries) % manifest_interval == 0:
            _write_manifest(
                output_dir=output_dir,
                query=query,
                entries=entries,
                complete=False,
            )
        if len(entries) % progress_interval == 0:
            elapsed = time.monotonic() - started_at
            downloaded_count = len(entries) - starting_count
            images_per_second = downloaded_count / elapsed
            remaining_seconds = (max_images - len(entries)) / images_per_second
            print(
                f"Progress: {len(entries)}/{max_images} images, "
                f"{candidates_seen} candidates, {images_per_second:.2f} images/s, "
                f"ETA {_format_duration(remaining_seconds)}",
                flush=True,
            )
        if len(entries) >= max_images:
            break

    _write_manifest(
        output_dir=output_dir,
        query=query,
        entries=entries,
        complete=len(entries) >= max_images,
    )
    return entries


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Tennis Court")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "data/dino_ssl/wikimedia_tennis_court",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--min-dimension", type=int, default=256)
    parser.add_argument("--min-color-difference", type=float, default=2.0)
    parser.add_argument("--thumbnail-width", type=int, default=512)
    parser.add_argument("--output-max-dimension", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--api-request-delay", type=float, default=0.2)
    parser.add_argument("--image-request-delay", type=float, default=0.5)
    parser.add_argument("--manifest-interval", type=int, default=25)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    resume_state = None
    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        elif args.resume:
            resume_state = _load_resume_state(args.output_dir)
        else:
            raise FileExistsError(
                f'Output directory already exists: "{args.output_dir}". '
                "Pass --overwrite to replace it or --resume to continue it."
            )
    elif args.resume:
        raise FileNotFoundError(f'Resume output directory does not exist: "{args.output_dir}"')

    client = WikimediaCommonsClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        retries=args.retries,
        api_request_delay=args.api_request_delay,
        image_request_delay=args.image_request_delay,
        thumbnail_width=args.thumbnail_width,
    )
    entries = build_dataset(
        client=client,
        query=args.query,
        output_dir=args.output_dir,
        max_images=args.max_images,
        candidate_multiplier=args.candidate_multiplier,
        min_dimension=args.min_dimension,
        min_color_difference=args.min_color_difference,
        output_max_dimension=args.output_max_dimension,
        jpeg_quality=args.jpeg_quality,
        manifest_interval=args.manifest_interval,
        progress_interval=args.progress_interval,
        resume_state=resume_state,
    )
    if len(entries) < args.max_images:
        raise RuntimeError(
            f"Only {len(entries)} of {args.max_images} requested images were available"
        )
    print(f"Downloaded {len(entries)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
