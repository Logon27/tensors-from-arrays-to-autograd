import gzip
import struct
import tempfile
import urllib.request
from pathlib import Path

from nn.array_backend import xp

# Automatically finds the correct temp folder for Windows/Linux/Mac
_DATA_DIR = Path(tempfile.gettempdir()) / "tensor_lib_data"

def _download(url: str, filepath: Path):
    """Download a URL to a local file path if it doesn't already exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        print(f"Downloading {url} to {filepath}")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

def _one_hot(x: xp.ndarray, k: int, dtype=xp.float32) -> xp.ndarray:
    """Create a one-hot encoding of x of size k."""
    return xp.array(x[:, None] == xp.arange(k), dtype)

def _flatten_images(x: xp.ndarray) -> xp.ndarray:
    """Flatten all but the first dimension of an ndarray."""
    return x.reshape((x.shape[0], -1))

# ==============================================================================
# MNIST / FASHION MNIST SHARED LOGIC
# ==============================================================================

def _parse_idx_labels(filepath: Path) -> xp.ndarray:
    """Parse labels from the IDX file format."""
    with gzip.open(filepath, "rb") as fh:
        struct.unpack(">II", fh.read(8))
        return xp.frombuffer(fh.read(), dtype=xp.uint8)

def _parse_idx_images(filepath: Path) -> xp.ndarray:
    """Parse images from the IDX file format."""
    with gzip.open(filepath, "rb") as fh:
        _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        return xp.frombuffer(fh.read(), dtype=xp.uint8).reshape(num_data, rows, cols)

def _load_idx_dataset(base_url: str, prefix: str = ""):
    """Downloads and parses an IDX formatted dataset (MNIST style)."""
    files = {
        "train_images": f"{prefix}train-images-idx3-ubyte.gz",
        "train_labels": f"{prefix}train-labels-idx1-ubyte.gz",
        "test_images": f"{prefix}t10k-images-idx3-ubyte.gz",
        "test_labels": f"{prefix}t10k-labels-idx1-ubyte.gz",
    }
    
    paths = {}
    for key, filename in files.items():
        # Using the standard filename for the URL, but allowing custom local prefix
        url_filename = filename.replace(prefix, "") 
        filepath = _DATA_DIR / filename
        _download(base_url + url_filename, filepath)
        paths[key] = filepath

    return (
        _parse_idx_images(paths["train_images"]),
        _parse_idx_labels(paths["train_labels"]),
        _parse_idx_images(paths["test_images"]),
        _parse_idx_labels(paths["test_labels"])
    )

def _process_image_dataset(train_imgs, train_lbls, test_imgs, test_lbls):
    """Normalizes, flattens, and one-hot encodes image datasets."""
    # Normalize pixel values to [0, 1] and flatten images from (N, H, W) to (N, H*W). E.g., (60000, 28, 28) -> (60000, 784)
    train_imgs = _flatten_images(train_imgs) / xp.float32(255.0)
    test_imgs = _flatten_images(test_imgs) / xp.float32(255.0)

    # One-hot encode the labels. E.g., 3 being the correct label -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    train_lbls = _one_hot(train_lbls, 10)
    test_lbls = _one_hot(test_lbls, 10)

    return train_imgs, train_lbls, test_imgs, test_lbls

# ==============================================================================
# DATASET APIs
# ==============================================================================

# Human-readable class labels for each dataset (index-aligned with the one-hot encoding).
MNIST_CLASS_NAMES = [str(i) for i in range(10)]

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def mnist():
    """Download, parse and process MNIST data."""
    print("Loading MNIST Dataset...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    raw_data = _load_idx_dataset(base_url)
    processed = _process_image_dataset(*raw_data)
    print("Finished loading MNIST Dataset.")
    return processed, MNIST_CLASS_NAMES

def fashion_mnist():
    """Download, parse and process Fashion MNIST data."""
    print("Loading FASHION MNIST Dataset...")
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    raw_data = _load_idx_dataset(base_url, prefix="fashion-")
    processed = _process_image_dataset(*raw_data)
    print("Finished loading FASHION MNIST Dataset.")
    return processed, FASHION_MNIST_CLASS_NAMES

