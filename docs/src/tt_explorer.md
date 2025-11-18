# Explorer

Explorer is an interactive GUI tool from [TT-MLIR](https://github.com/tenstorrent/tt-mlir) for visualizing and experimenting with model graphs (including Tenstorrent's MLIR dialects), compiling and executing your model on Tenstorrent hardware.

## What is Explorer?

Explorer is a visual debugging and performance analysis tool that allows you to:
- **Visualize MLIR graphs**: Inspect your model graph with hierarchical visualization
- **Compile and execute your model**: Compile your model to Tenstorrent hardware and execute it
- **Debug performance**: Identify bottlenecks and see affects of optimizations on runtime performance

## Building with Explorer

Explorer is only available when building TT-XLA from source. It is not included in pre-built wheels. It is **disabled by default** in TT-XLA. You can enable it by building with the `TTXLA_ENABLE_EXPLORER` CMake option:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DTTXLA_ENABLE_EXPLORER=ON
cmake --build build
```

> **Note:** Enabling Explorer also enables Tracy performance tracing (`TTMLIR_ENABLE_PERF_TRACE`), which may slow down execution. For production deployments or performance benchmarking, consider building with `-DTTXLA_ENABLE_EXPLORER=OFF`.

## Using Explorer

After building with Explorer enabled, launch the tool by running:

```bash
tt-explorer
```

This will start the interactive GUI for analyzing your model's compilation and execution.

### Example graph to try out
```mlir
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = ttir.empty() : tensor<64x128xbf16>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %4 = ttir.empty() : tensor<64x128xbf16>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %6 = ttir.empty() : tensor<64x128xbf16>
    %7 = "ttir.relu"(%5, %6) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %7 : tensor<64x128xbf16>
  }
}
```

### View graphs from tests

You can use the EXPLORER_EXPORT_LEVEL environment variable to collect graphs being compiled from a pytest.

#### Export Usage
Graphs will be saved to `~/explorer/` with organized subdirectories by test name and compilation stage.
```bash
export EXPLORER_EXPORT_LEVEL=pass  # or "once", "pipeline", "transformation"
pytest your_test.py
```

#### Dump Levels
- **`once`**: Export IR only once after each major compilation stage
- **`pipeline`**: Export IR after each pipeline boundary
- **`pass`**: Export IR after each compiler pass
- **`transformation`**: Export IR after every transformation (most verbose)

#### Viewing Exported Graphs

Once exported, you can preload graphs saved under `~/explorer/` in the Explorer GUI.
You can also open the `.mlir` files directly to view the graph:
```bash
tt-explorer ~/explorer/your_test_name/compilation_stage/0_initial.mlir
```
## Learn More

For detailed documentation on how to use Explorer, including tutorials and advanced features, see the [TT-MLIR Explorer Documentation](https://docs.tenstorrent.com/tt-mlir/tt-explorer/tt-explorer.html).

Explorer is based on [Google's Model Explorer](https://github.com/google-ai-edge/model-explorer) with added support for Tenstorrent hardware compilation and execution.
