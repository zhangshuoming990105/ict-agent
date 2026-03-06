#!/usr/bin/env python
"""Correctness verification for CUDA extension model."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from model import Model, get_init_inputs, get_inputs
from model_new import ModelNew


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(item, fn) for item in tensors]
    if isinstance(tensors, dict):
        return {key: transform_tensors(value, fn) for key, value in tensors.items()}
    return tensors


def check_equal(actual, expected):
    assert type(actual) is type(expected), f"{type(actual)=} != {type(expected)=}"
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), f"{len(actual)=} != {len(expected)=}"
        for left, right in zip(actual, expected):
            check_equal(left, right)
    elif isinstance(actual, dict):
        for key, value in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            check_equal(actual[key], value)
    elif isinstance(actual, (str, float, int)):
        assert actual == expected, f"{actual=} != {expected=}"
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    else:
        raise TypeError(f"Unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    excludes = excludes or set()
    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function torch.nn.functional.{__name} is not allowed in this context."
                )

            setattr(F, name, wrapper)
    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def initialize_models():
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]
    torch_model = Model(*init_inputs).eval().cuda()
    cuda_model = ModelNew(*init_inputs).eval().cuda()
    cuda_model.load_state_dict(torch_model.state_dict())
    return torch_model, cuda_model


def build_inputs():
    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda tensor: tensor.cuda())
    cuda_inputs = transform_tensors(torch_inputs, lambda tensor: tensor.clone())
    return torch_inputs, cuda_inputs


def main():
    torch_model, cuda_model = initialize_models()
    with torch.no_grad():
        for index in range(5):
            torch_inputs, cuda_inputs = build_inputs()
            torch_output = torch_model(*torch_inputs)
            with block_torch_functional():
                cuda_output = cuda_model(*cuda_inputs)
            check_equal(cuda_output, torch_output)
            print(f"[PASS] check {index + 1}/5")
    torch.cuda.synchronize()
    print("[PASS] verify success")


if __name__ == "__main__":
    main()
