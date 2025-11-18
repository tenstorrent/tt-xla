import ttmlir
from ttmlir.ir import *
from ttmlir.dialects import ttir, func, ttcore, tensor
from collections import defaultdict

def walk_operations(op, depth=0):
    """Recursively walk all operations in the IR."""
    indent = "  " * depth
    print(f"{indent}Op: {op.name}")

    # Walk regions
    for region in op.regions:
        for block in region:
            for child_op in block.operations:
                walk_operations(child_op, depth + 1)

def check_ttir(ttir_ir_path: str, expected_ops) -> bool:
    with open(ttir_ir_path, 'r') as f:
        ir_content = f.read()

    try:
        with Context() as ctx:
            module = Module.parse(ir_content)

            print("="*80)
            print("Walking all operations in TTIR:")
            print("="*80)

            # Walk and print all operations
            for op in module.body.operations:
                walk_operations(op)

    except Exception as e:
        print("Failed to parse TTIR MLIR due to exception:", e)
        import traceback
        traceback.print_exc()

def check_ttnn(ttnn_ir_path:str, expected_ops) -> bool:
    with open(ttnn_ir_path, 'r') as f:
        ir_content = f.read()
    pass



check_ttir("output_artifact/test_llama_step[LLMRunMode.PREFILL]_ttir.mlir", ["hi"])
# check_ttnn("output_artifact/test_llama_step[LLMRunMode.PREFILL]_ttnn.mlir", ["hi"])