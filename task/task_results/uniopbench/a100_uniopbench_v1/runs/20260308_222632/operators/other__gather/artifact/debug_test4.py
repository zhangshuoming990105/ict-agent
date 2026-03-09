"""Debug test to understand tensor generation."""

import torch
from optest.tools.checker import SEED
from dataclasses import dataclass


@dataclass
class GatherParams:
    """Gather test parameters."""
    data_shape: tuple = (50, 128, 4)
    num_indices: int = 4
    output_shape: tuple = (50, 128, 4)


def resolve_shape(shape_spec, params):
    """Resolve a shape specification."""
    result = []
    i = 0
    while i < len(shape_spec):
        item = shape_spec[i]
        if isinstance(item, str):
            # It's a parameter name
            attr = getattr(params, item)
            if isinstance(attr, tuple):
                result.extend(attr)
            else:
                result.append(attr)
        elif isinstance(item, int):
            # Direct integer - might be an index into next param
            if i + 1 < len(shape_spec) and isinstance(shape_spec[i+1], str):
                # This int is actually an index
                i += 1
                param_name = shape_spec[i]
                attr = getattr(params, param_name)
                if isinstance(attr, tuple):
                    result.append(attr[item])
                else:
                    result.append(attr)
            else:
                result.append(item)
        i += 1
    return tuple(result)


params = GatherParams()

# Test different shape specs
specs = [
    ("data_shape",),
    ("num_indices",),
    ("data_shape", 0, "data_shape", 1, "num_indices"),
]

for spec in specs:
    resolved = resolve_shape(spec, params)
    print(f"Spec: {spec}")
    print(f"Resolved: {resolved}")
    print()
