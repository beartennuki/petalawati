import zipfile
from pathlib import Path
from prefect import task
from app.config import UPLOADS_DIR


@task(name="extract-dataset")
def extract_dataset(zip_path: str, job_id: str) -> dict:
    """Extract ZIP to uploads/{job_id}/ and validate ImageFolder structure."""
    dest = UPLOADS_DIR / job_id
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Handle single nested folder (common when zipping a folder itself)
    top_level = list(dest.iterdir())
    if len(top_level) == 1 and top_level[0].is_dir():
        actual_root = top_level[0]
    else:
        actual_root = dest

    classes = sorted(
        p.name for p in actual_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not classes:
        raise ValueError("ZIP must contain class subdirectories (e.g. cats/, dogs/)")

    num_images = sum(
        1 for cls in classes
        for f in (actual_root / cls).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    return {"data_dir": str(actual_root), "classes": classes, "num_images": num_images}


@task(name="build-datasets")
def build_datasets(data_dir: str, image_size: int, batch_size: int, val_split: float, augment: bool):
    """Return (train_ds, val_ds, class_names) using tf.keras utilities."""
    import tensorflow as tf

    img_size = (image_size, image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
