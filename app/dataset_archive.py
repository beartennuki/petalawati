import shutil
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_ignored_part(part: str) -> bool:
    return part == "__MACOSX" or part.startswith("._") or part.startswith(".")


def _visible_parts(name: str) -> tuple[str, ...] | None:
    parts = tuple(part for part in PurePosixPath(name).parts if part not in {"", "."})
    if not parts or any(_is_ignored_part(part) for part in parts):
        return None
    return parts


def _normalized_members(names: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    members = []
    for name in names:
        parts = _visible_parts(name)
        if parts and not name.endswith("/"):
            members.append((name, parts))

    if not members:
        return []

    first_segments = {parts[0] for _, parts in members}
    if len(first_segments) == 1 and all(len(parts) >= 2 for _, parts in members):
        return [(name, parts[1:]) for name, parts in members]

    return members


def inspect_dataset_archive(names: list[str]) -> tuple[list[str], int]:
    normalized_members = _normalized_members(names)
    image_members = [
        parts for _, parts in normalized_members
        if len(parts) >= 2 and Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS
    ]
    classes = sorted({parts[0] for parts in image_members})
    return classes, len(image_members)


def extract_dataset_archive(zip_path: str | Path, dest: Path) -> tuple[Path, list[str], int]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        normalized_members = _normalized_members(zf.namelist())
        image_members = [
            (name, parts) for name, parts in normalized_members
            if (
                len(parts) >= 2
                and Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS
                and zf.getinfo(name).file_size > 0
            )
        ]

        for original_name, parts in image_members:
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(original_name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    classes, num_images = inspect_dataset_archive([str(Path(*parts)) for _, parts in image_members])
    if not classes:
        raise ValueError("ZIP must contain class subdirectories (e.g. cats/, dogs/)")

    return dest, classes, num_images
