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


def validate_dataset_archive(zip_path: str | Path) -> list[str]:
    """
    Fast structural checks run at upload time.
    Returns human-readable error strings; empty list = valid.

    Checks:
    1. Structure — every file must sit at class/filename depth.
    2. File type — every visible file must carry a supported image extension.

    Channel consistency is a separate, user-toggleable check run at job
    submission time via check_channel_consistency().
    """
    errors: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        normalized = _normalized_members(zf.namelist())

        bad_structure = ["/".join(parts) for _, parts in normalized if len(parts) != 2]
        non_image = [
            "/".join(parts)
            for _, parts in normalized
            if len(parts) == 2 and Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS
        ]

        if bad_structure:
            sample = bad_structure[:3]
            tail = " ..." if len(bad_structure) > 3 else ""
            errors.append(
                f"Files must be placed inside a class folder at exactly one level deep "
                f"(e.g. cats/img.jpg). {len(bad_structure)} bad path(s): "
                f"{', '.join(sample)}{tail}"
            )

        if non_image:
            sample = non_image[:3]
            tail = " ..." if len(non_image) > 3 else ""
            allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
            errors.append(
                f"{len(non_image)} non-image file(s) found: {', '.join(sample)}{tail}. "
                f"Allowed extensions: {allowed}"
            )

    return errors


def check_channel_consistency(zip_path: str | Path) -> list[str]:
    """
    Scans every image in the ZIP for TF-compatible channel modes.
    Returns human-readable error strings; empty list = all images are compatible.
    """
    from PIL import Image

    # TensorFlow decode_image with color_mode="rgb" (channels=3) handles:
    #   L (1ch)  → replicated to 3ch
    #   P        → palette expanded to RGB
    #   RGB      → used as-is
    #   RGBA     → alpha dropped
    # It cannot handle LA (2ch grayscale+alpha) and raises InvalidArgumentError.
    _TF_SAFE_MODES = {"L", "RGB", "RGBA", "P"}

    bad_modes: list[str] = []
    unreadable: list[str] = []
    class_first_channel: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        normalized = _normalized_members(zf.namelist())

        for name, parts in normalized:
            if len(parts) != 2 or Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with zf.open(name) as f:
                    with Image.open(f) as img:
                        img.load()
                        mode = img.mode
            except Exception:
                unreadable.append("/".join(parts))
                continue

            if mode not in _TF_SAFE_MODES:
                bad_modes.append(f"{'/'.join(parts)} (mode='{mode}')")
            else:
                cls = parts[0]
                if cls not in class_first_channel:
                    class_first_channel[cls] = "grayscale" if mode == "L" else "color"

    errors: list[str] = []

    if bad_modes:
        sample = bad_modes[:3]
        tail = " ..." if len(bad_modes) > 3 else ""
        errors.append(
            f"{len(bad_modes)} image(s) with unsupported channel format "
            f"(TensorFlow requires 1, 3, or 4 channels). Remove or convert these files: "
            f"{', '.join(sample)}{tail}. "
            f"Common cause: grayscale+alpha (LA) PNG — convert to RGB or L before uploading."
        )

    if unreadable:
        sample = unreadable[:3]
        tail = " ..." if len(unreadable) > 3 else ""
        errors.append(
            f"{len(unreadable)} image(s) could not be fully decoded during consistency check. "
            f"Remove or re-export these files: {', '.join(sample)}{tail}."
        )

    unique_channels = set(class_first_channel.values())
    if not bad_modes and not unreadable and len(unique_channels) > 1:
        detail = ", ".join(f"{c}={m}" for c, m in sorted(class_first_channel.items()))
        errors.append(
            f"Inconsistent image channels across classes ({detail}). "
            "All classes must be either colour or grayscale."
        )

    return errors


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
