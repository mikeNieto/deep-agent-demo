from __future__ import annotations

import subprocess
from pathlib import Path
from typing import BinaryIO

from app.storage.files import ensure_parent


def convert_wav_to_mp3(source: Path, destination: Path) -> Path:
    ensure_parent(destination)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    return destination


def save_upload(file_obj: BinaryIO, destination: Path) -> Path:
    ensure_parent(destination)
    with destination.open("wb") as handle:
        handle.write(file_obj.read())
    return destination
