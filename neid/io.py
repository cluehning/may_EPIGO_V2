from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_gz_track(path: Path) -> np.ndarray:
    """
    Load a gzipped track file into a NumPy array.

    Supports:
    - single-column numeric files
    - multi-column numeric files (fallback: last column)

    Parameters
    ----------
    path : Path
        Path to the gzipped input file.

    Returns
    -------
    np.ndarray
        Loaded numeric data.
    """
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        try:
            data = np.loadtxt(file_obj)
        except ValueError:
            file_obj.seek(0)
            data = np.loadtxt(file_obj, usecols=[-1])

    return np.asarray(data)


def load_all_gz_tracks(data_dir: Path) -> dict[str, np.ndarray]:
    """
    Load all `.gz` track files from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing gzipped track files.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from filename (without `.gz`) to loaded NumPy array.
    """
    tracks: dict[str, np.ndarray] = {}

    for file_path in data_dir.rglob("*.gz"):
        try:
            data = load_gz_track(file_path)
            name = file_path.name.removesuffix(".gz")
            tracks[name] = data
        except OSError as exc:
            print(f"Skipping {file_path}: {exc}")
        except ValueError as exc:
            print(f"Skipping {file_path}: {exc}")

    return tracks


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _missing(dep: str, extra: str = "") -> ImportError:
    """
    Build a helpful ImportError for optional dependencies.

    Parameters
    ----------
    dep : str
        Dependency name.
    extra : str, optional
        Extra installation guidance.

    Returns
    -------
    ImportError
        Configured import error.
    """
    msg = f"Missing dependency '{dep}'. Install it with: pip install {dep}"
    if extra:
        msg = f"{msg}{extra}"
    return ImportError(msg)


def read_yaml(path: Path) -> dict[str, Any]:
    """
    Read a YAML file into a dictionary.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _missing("pyyaml") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    """
    Read a JSON file into a dictionary.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object, got {type(data)}")
    return data


def write_json(path: Path, obj: Mapping[str, Any], indent: int = 2) -> None:
    """
    Write a dictionary-like object to JSON.

    Parameters
    ----------
    path : Path
        Output path.
    obj : Mapping[str, Any]
        Data to serialize.
    indent : int, optional
        JSON indentation level.
    """
    ensure_dir(path.parent)
    text = json.dumps(obj, indent=indent, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> Any:
    """
    Read a CSV file via pandas.

    Parameters
    ----------
    path : Path
        Input CSV path.

    Returns
    -------
    Any
        Pandas DataFrame.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise _missing("pandas") from exc
    return pd.read_csv(path)


def write_csv(path: Path, df: Any) -> None:
    """
    Write a pandas DataFrame to CSV.

    Parameters
    ----------
    path : Path
        Output CSV path.
    df : Any
        Pandas DataFrame-like object.
    """
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def read_parquet(path: Path) -> Any:
    """
    Read a Parquet file via pandas.

    Parameters
    ----------
    path : Path
        Input Parquet path.

    Returns
    -------
    Any
        Pandas DataFrame.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise _missing("pandas") from exc

    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise _missing(
            "pyarrow",
            " (recommended) or pip install fastparquet",
        ) from exc


def write_parquet(path: Path, df: Any) -> None:
    """
    Write a pandas DataFrame to Parquet.

    Parameters
    ----------
    path : Path
        Output Parquet path.
    df : Any
        Pandas DataFrame-like object.
    """
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise _missing(
            "pyarrow",
            " (recommended) or pip install fastparquet",
        ) from exc


def read_npy(path: Path) -> np.ndarray:
    """
    Read a NumPy `.npy` file.

    Parameters
    ----------
    path : Path
        Input path.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    return np.load(path)


def write_npy(path: Path, arr: np.ndarray) -> None:
    """
    Write a NumPy array to `.npy`.

    Parameters
    ----------
    path : Path
        Output path.
    arr : np.ndarray
        Array to save.
    """
    ensure_dir(path.parent)
    np.save(path, arr)
