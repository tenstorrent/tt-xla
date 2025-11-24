## Troubleshooting

### Code Generation Fails

**Symptom:**
```
ERROR: tt-alchemist generatePython failed
```

**Cause:** Code generation process encountered an error

**Solutions:**

1. **Check export path is writable:**
   ```bash
   mkdir -p <export_path>
   touch <export_path>/test && rm <export_path>/test
   ```

2. **Verify TTIR was generated:**
   ```bash
   ls -lh <export_path>/ttir.mlir
   ```
   If `ttir.mlir` is missing or empty (0 bytes), compilation failed before code generation.

3. **Check for compilation errors:**
   Review the full output for errors before the "generatePython failed" message.

4. **Try with minimal model:**
   Test with a simple model to isolate the issue:
   ```python
   class MinimalModel(torch.nn.Module):
       def forward(self, x):
           return x + 1
   ```

---

### Export Path Not Set

**Symptom:**
```
Compile option 'export_path' must be provided when backend is not 'TTNNFlatbuffer'
```

**Cause:** The `export_path` option is missing

**Solution:** Add `export_path` to your compiler options:

```python
options = {
    "backend": "codegen_py",
    "export_path": "./output"  # ← Add this
}
```

---

### Generated Code Execution Fails

**Symptom:** Errors when running generated Python code via `./run`

**Possible Causes & Solutions:**

1. **TT-XLA not built:**
   ```bash
   cd /path/to/tt-xla
   cmake --build build
   ```

2. **Hardware not accessible:**
   ```bash
   tt-smi  # Should show your Tenstorrent devices
   ```

3. **Wrong hardware configuration:**
   - Verify generated code matches your hardware setup
   - Check device IDs and chip counts
   - Rebuild TT-XLA if hardware configuration changed

4. **Missing dependencies:**
   ```bash
   source venv/activate  # Ensure virtual environment is active
   ```

---

### Generated C++ Code Won't Compile

**Symptom:** C++ compilation errors in generated code

**Solutions:**

1. **Check TT-NN headers are available:**
   ```bash
   find /opt/ttmlir-toolchain -name "ttnn*.h"
   ```

2. **Verify C++ compiler version:**
   Generated code requires C++17 or later

3. **Link against TT-NN library:**
   Ensure your build system links the TT-NN library correctly

---
