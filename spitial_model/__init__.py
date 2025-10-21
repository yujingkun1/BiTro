"""spitial_model package."""

# Setup path directly without circular import
import os
import sys

_current_file_dir = os.path.dirname(
    os.path.abspath(__file__))  # spitial_model directory
_cell2gene_dir = os.path.dirname(_current_file_dir)  # Cell2Gene directory

if _cell2gene_dir not in sys.path:
    sys.path.insert(0, _cell2gene_dir)

_cwd = os.getcwd()
if _cwd != _cell2gene_dir and _cwd not in sys.path:
    sys.path.insert(0, _cwd)
