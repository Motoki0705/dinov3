#!/usr/bin/env python3
"""Download a small color-image dataset from Wikimedia Commons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError

API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "tennis-lab-dino-ssl/1.0 "
    "(https://github.com/Motoki0705/tennis-lab; Wikimedia dataset builder)"
)
SUPPORTED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


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


class CommonsClient(Protocol):
    def search_files(self, query: str, limit: int) -> Iterable[CommonsFile]: ...

    def download(self, url: str) -> bytes: ...


class WikimediaCommonsClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 30.0,
        retries: int = 3,
        request_delay: float = 0.1,
        thumbnail_width: int = 1600,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A descriptive Wikimedia User-Agent is required")
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self.thumbnail_width = thumbnail_width

    def _request(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return payload
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2**attempt
            except URLError:
                if attempt >= self.retries:
                    raise
                delay = 2**attempt
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_URL}?{urlencode(params)}"
        return json.loads(self._request(url))

    def search_files(self, query: str, limit: int) -> Iterable[CommonsFile]:
        continuation: dict[str, Any] = {}
        yielded = 0
        while yielded < limit:
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": f'intitle:"{query}"',
                "gsrnamespace": 6,
                "gsrlimit": min(50, limit - yielded),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": self.thumbnail_width,
                **continuation,
            }
            result = self._request_json(params)
            for page in result.get("query", {}).get("pages", []):
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
                    page_id=int(page["pageid"]),
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
                return

    def download(self, url: str) -> bytes:
        return self._request(url)


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
    return image, color_difference


def _metadata_value(file: CommonsFile, key: str) -> str:
    return file.metadata.get(key, "")


def build_dataset(
    *,
    client: CommonsClient,
    query: str,
    output_dir: Path,
    max_images: int,
    candidate_multiplier: int,
    min_dimension: int,
    min_color_difference: float,
    jpeg_quality: int,
) -> list[ManifestEntry]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    seen_hashes: set[str] = set()
    candidate_limit = max(max_images, max_images * candidate_multiplier)

    for commons_file in client.search_files(query, candidate_limit):
        if commons_file.mime not in SUPPORTED_MIME_TYPES:
            continue
        try:
            data = client.download(commons_file.download_url)
        except (HTTPError, URLError, TimeoutError):
            continue
        prepared = _prepare_image(
            data,
            min_dimension=min_dimension,
            min_color_difference=min_color_difference,
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
        if len(entries) >= max_images:
            break

    manifest = {
        "source": "Wikimedia Commons",
        "query": query,
        "num_images": len(entries),
        "images": [asdict(entry) for entry in entries],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--thumbnail-width", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Output directory already exists: "{args.output_dir}". '
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    client = WikimediaCommonsClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
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
        jpeg_quality=args.jpeg_quality,
    )
    if not entries:
        raise RuntimeError("No eligible color images were downloaded")
    print(f"Downloaded {len(entries)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
