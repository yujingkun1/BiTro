from .lora import LoRALinear, LoRAMultiheadSelfAttention, apply_lora_to_linear_modules

# Import OptimizedTransformerPredictor from the parent models.py module
# Since models.py and models/ directory are at the same level, we need to import it differently
import sys
import os
import importlib.util

# Get the parent directory (bulk_model/)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_models_py_path = os.path.join(_parent_dir, 'models.py')

if os.path.exists(_models_py_path):
    # Load models.py as a module
    spec = importlib.util.spec_from_file_location("bulk_model_models_py", _models_py_path)
    _models_py_module = importlib.util.module_from_spec(spec)
    # Add parent directory to path for relative imports in models.py
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    spec.loader.exec_module(_models_py_module)
    OptimizedTransformerPredictor = _models_py_module.OptimizedTransformerPredictor
    StaticGraphGNN = getattr(_models_py_module, 'StaticGraphGNN', None)
else:
    raise ImportError(f"Cannot find models.py at {_models_py_path}")

__all__ = ['LoRALinear', 'LoRAMultiheadSelfAttention', 'apply_lora_to_linear_modules', 
           'OptimizedTransformerPredictor']
if StaticGraphGNN is not None:
    __all__.append('StaticGraphGNN')

