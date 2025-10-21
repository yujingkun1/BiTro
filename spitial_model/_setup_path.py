"""Setup path for spitial_model imports.

This module ensures that the Cell2Gene directory is in sys.path,
allowing proper imports from spitial_model and its submodules.
"""

import os
import sys

# Get the Cell2Gene directory (parent of spitial_model directory)
_current_file_dir = os.path.dirname(
    os.path.abspath(__file__))  # spitial_model directory
_cell2gene_dir = os.path.dirname(_current_file_dir)  # Cell2Gene directory

# Add Cell2Gene to sys.path if not already present
if _cell2gene_dir not in sys.path:
    sys.path.insert(0, _cell2gene_dir)
