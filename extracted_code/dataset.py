"""Backward-compatible CLI wrapper for the pixel-wise NDWI dataset generator.

The reusable implementation lives in src.dataset_generator. This file remains
importable from old scripts; it contains no notebook-only commands.
"""
from src.dataset_generator import main, process_ndwi_tif

if __name__ == "__main__":
    main()
