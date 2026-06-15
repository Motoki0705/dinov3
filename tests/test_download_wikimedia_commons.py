import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.download_wikimedia_commons import (
    AdaptiveRateLimiter,
    CommonsFile,
    WikimediaCommonsClient,
    _load_resume_state,
    build_dataset,
)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, 240), color).save(output, format="PNG")
    return output.getvalue()


class FakeCommonsClient:
    def __init__(self):
        self.files = [
            CommonsFile(
                page_id=1,
                title="File:Grayscale.png",
                download_url="https://example.test/grayscale.png",
                description_url="https://example.test/wiki/grayscale",
                mime="image/png",
                width=320,
                height=240,
                metadata={},
            ),
            CommonsFile(
                page_id=2,
                title="File:Tennis court.png",
                download_url="https://example.test/tennis.png",
                description_url="https://example.test/wiki/tennis",
                mime="image/png",
                width=320,
                height=240,
                metadata={
                    "LicenseShortName": "CC BY-SA 4.0",
                    "LicenseUrl": "https://creativecommons.org/licenses/by-sa/4.0",
                    "Artist": "Example author",
                },
            ),
        ]
        self.payloads = {
            self.files[0].download_url: _image_bytes((128, 128, 128)),
            self.files[1].download_url: _image_bytes((20, 160, 60)),
        }

    def search_files(self, query: str, limit: int):
        assert query == "Tennis Court"
        return iter(self.files[:limit])

    def download(self, url: str) -> bytes:
        return self.payloads[url]


def test_build_dataset_filters_grayscale_and_writes_attribution(tmp_path: Path):
    entries = build_dataset(
        client=FakeCommonsClient(),
        query="Tennis Court",
        output_dir=tmp_path,
        max_images=1,
        candidate_multiplier=2,
        min_dimension=128,
        min_color_difference=2.0,
        output_max_dimension=256,
        jpeg_quality=90,
    )

    assert len(entries) == 1
    assert entries[0].commons_page_id == 2
    assert entries[0].license == "CC BY-SA 4.0"
    image_path = tmp_path / entries[0].filename
    assert image_path.is_file()
    assert entries[0].sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    with Image.open(image_path) as image:
        assert max(image.size) <= 256

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["num_images"] == 1
    assert manifest["images"][0]["artist"] == "Example author"


def test_wikimedia_search_starts_with_exact_phrase(monkeypatch):
    client = WikimediaCommonsClient(user_agent="test/1.0 (test@example.com)")
    captured = []

    def fake_request_json(params):
        captured.append(params)
        return {}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert list(client.search_files("Tennis Court", limit=1)) == []
    assert captured[0]["gsrsearch"] == '"Tennis Court"'
    assert any(params["gsrsearch"] == "Tennis Court" for params in captured)


def test_resume_state_reuses_existing_images(tmp_path: Path):
    entries = build_dataset(
        client=FakeCommonsClient(),
        query="Tennis Court",
        output_dir=tmp_path,
        max_images=1,
        candidate_multiplier=2,
        min_dimension=128,
        min_color_difference=2.0,
        output_max_dimension=256,
        jpeg_quality=90,
    )
    resumed = _load_resume_state(tmp_path)

    assert resumed.entries == entries
    assert resumed.page_ids == {2}
    assert len(resumed.pixel_hashes) == 1


def test_retry_after_does_not_become_the_sustained_request_delay():
    limiter = AdaptiveRateLimiter(initial_delay=0.5, min_delay=0.25, max_delay=60)

    retry_wait = limiter.record_throttle(retry_after=11)

    assert retry_wait == 11
    assert limiter.delay == 0.625
