import os
import sys
import argparse
from dataclasses import dataclass

import torch
from optest import (
    TestCase,
    TensorSpec,
    ScalarSpec,
    TestConfig,
    CUDABackend,
    TritonBackend,
    run_test,
    print_result,
    create_suite,
    print_suite_summary,
)

from optest.execution.benchmark import benchmark_kernel
from optest.core.result import TestResult

TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)
from torch_.ref import torch_kernel

@dataclass
class NmsParams:
    num_boxes: int = 2048
    iou_threshold: float = 0.5
    dtype: torch.dtype = torch.float32

    @property
    def total_elements(self):
        return self.num_boxes * 4

testcase = TestCase(
    tensor_specs=[
        TensorSpec("boxes", "dtype", ("num_boxes", 4), "input"),
        TensorSpec("keep", torch.int32, ("num_boxes",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("num_boxes", int, lambda p: p.num_boxes),
        ScalarSpec("iou_threshold", float, lambda p: p.iou_threshold),
    ],
    torch_kernel=torch_kernel,
)

def run_nms_logic(params, backend, config):
    device = "cuda" if backend.name == "CUDA" else "cpu"
    print(f"\n>> Running NMS: Boxes={params.num_boxes}, Thresh={params.iou_threshold}")

    boxes = torch.rand((params.num_boxes, 4), dtype=params.dtype, device=device)
    boxes[:, 2:] += boxes[:, :2] 
    
    scores = torch.rand(params.num_boxes, dtype=params.dtype, device=device)
    _, indices = scores.sort(descending=True)
    sorted_boxes = boxes[indices].contiguous()
    
    keep_mask = torch.ones(params.num_boxes, dtype=torch.int32, device=device)

    ref_inputs = [sorted_boxes, params.iou_threshold] 
    
    test_inputs = [sorted_boxes, keep_mask, params.num_boxes, params.iou_threshold]
    test_inputs_ptr = backend.prepare_inputs(test_inputs)

    from optest.execution.runner import TestRunner
    runner = TestRunner(TESTCASE_DIR, backend)
    
    output_ref = runner.ref_backend.execute(testcase.torch_kernel, ref_inputs)
    
    kernel = backend.load_kernel(TESTCASE_DIR, testcase.get_argtypes())
    backend.execute(kernel, test_inputs_ptr)
    torch.cuda.synchronize()
    
    # 自定义验证逻辑 (允许 <1% 的误差)
    diff = torch.abs(output_ref - keep_mask)
    mismatches = diff.sum().item()
    error_rate = mismatches / params.num_boxes
    is_passed = error_rate < 0.01

    if config.verbose:
        print(f"   Ref kept: {output_ref.sum().item()}, CUDA kept: {keep_mask.sum().item()}")
        print(f"   Mismatches: {mismatches} ({error_rate:.2%})")

    if is_passed:
        if config.enable_perf:
            print("   Running benchmarks...")
            
            ref_ms = benchmark_kernel(
                testcase.torch_kernel, ref_inputs, 
                warmup=config.perf_warmup, iterations=config.perf_iterations
            )
            
            test_ms = benchmark_kernel(
                kernel, test_inputs_ptr,
                warmup=config.perf_warmup, iterations=config.perf_iterations
            )
            
            speedup = ref_ms / test_ms if test_ms > 0 else 0
            print("-" * 80)
            print("PERFORMANCE:")
            print(f"  Reference: {ref_ms:.5f} ms")
            print(f"  Test:      {test_ms:.5f} ms")
            print(f"  Speedup:   {speedup:.2f}x 🚀")
            print("-" * 80)
        
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--variants", default="none")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = TestConfig(
        enable_perf=not args.no_perf,
        verbose=args.verbose,
        perf_warmup=100,
        perf_iterations=1000
    )
    backend = CUDABackend() if args.backend == "cuda" else TritonBackend()

    print(f"\n{'=' * 80}")
    print(f"Testing NMS Operator with {backend.name}")
    print(f"{'=' * 80}")

    if args.variants == "yaml":
        # 手动定义变种列表 (模拟 cases.yaml)
        variants = [
            NmsParams(num_boxes=1024, iou_threshold=0.5),
            NmsParams(num_boxes=2048, iou_threshold=0.5),
            NmsParams(num_boxes=4096, iou_threshold=0.5),
            NmsParams(num_boxes=8192, iou_threshold=0.7),
        ]
        all_passed = True
        for p in variants:
            if not run_nms_logic(p, backend, config):
                all_passed = False
        sys.exit(0 if all_passed else 1)
    else:
        # 单次运行默认参数
        p = NmsParams()
        passed = run_nms_logic(p, backend, config)
        sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()