"""Tests for content utility functions (image dimensions, encoding)."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

from loom.content_utils import (
    encode_file_base64,
    encode_image_base64,
    get_image_dimensions,
    get_image_dimensions_from_bytes,
)


class TestGetImageDimensionsFromBytes:
    def test_png(self):
        # Build a minimal PNG header with IHDR chunk
        header = b"\x89PNG\r\n\x1a\n"
        header += b"\x00" * 4  # chunk length
        header += b"IHDR"
        header += struct.pack(">II", 1920, 1080)  # width, height
        w, h = get_image_dimensions_from_bytes(header, ".png")
        assert w == 1920
        assert h == 1080

    def test_gif87a(self):
        header = b"GIF87a" + struct.pack("<HH", 320, 240) + b"\x00" * 20
        w, h = get_image_dimensions_from_bytes(header, ".gif")
        assert w == 320
        assert h == 240

    def test_gif89a(self):
        header = b"GIF89a" + struct.pack("<HH", 640, 480) + b"\x00" * 20
        w, h = get_image_dimensions_from_bytes(header, ".gif")
        assert w == 640
        assert h == 480

    def test_bmp(self):
        header = b"BM" + b"\x00" * 16 + struct.pack("<II", 800, 600) + b"\x00" * 6
        w, h = get_image_dimensions_from_bytes(header, ".bmp")
        assert w == 800
        assert h == 600

    def test_jpeg_returns_zero(self):
        header = b"\xff\xd8\xff\xe0" + b"\x00" * 28
        w, h = get_image_dimensions_from_bytes(header, ".jpg")
        assert w == 0
        assert h == 0

    def test_unknown_format(self):
        w, h = get_image_dimensions_from_bytes(b"random bytes", ".xyz")
        assert w == 0
        assert h == 0

    def test_short_header(self):
        w, h = get_image_dimensions_from_bytes(b"\x89PN", ".png")
        assert w == 0
        assert h == 0


class TestGetImageDimensions:
    def test_png_file(self, tmp_path: Path):
        # Create a minimal valid PNG header
        header = b"\x89PNG\r\n\x1a\n"
        header += b"\x00" * 4 + b"IHDR"
        header += struct.pack(">II", 256, 128)
        header += b"\x00" * 100

        img = tmp_path / "test.png"
        img.write_bytes(header)

        w, h = get_image_dimensions(img)
        assert w == 256
        assert h == 128

    def test_nonexistent_file(self, tmp_path: Path):
        w, h = get_image_dimensions(tmp_path / "missing.png")
        assert w == 0
        assert h == 0


class TestEncodeImageBase64:
    def test_small_image(self, tmp_path: Path):
        img = tmp_path / "test.png"
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img.write_bytes(data)

        result = encode_image_base64(img)
        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded == data

    def test_nonexistent_file(self, tmp_path: Path):
        result = encode_image_base64(tmp_path / "missing.png")
        assert result is None

    def test_too_large_no_pillow(self, tmp_path: Path, monkeypatch):
        img = tmp_path / "huge.png"
        # Create a file > 5MB
        img.write_bytes(b"\x89PNG" + b"\x00" * (6 * 1024 * 1024))

        # Mock out the resize function to simulate no Pillow
        import loom.content_utils as cu

        def mock_resize(*args, **kwargs):
            raise ImportError("no pillow")

        monkeypatch.setattr(cu, "_resize_and_encode", mock_resize)
        result = encode_image_base64(img)
        assert result is None


class TestEncodeFileBase64:
    def test_encode_file(self, tmp_path: Path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 test content")

        result = encode_file_base64(f)
        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded == b"%PDF-1.4 test content"

    def test_nonexistent_file(self, tmp_path: Path):
        result = encode_file_base64(tmp_path / "missing.pdf")
        assert result is None
