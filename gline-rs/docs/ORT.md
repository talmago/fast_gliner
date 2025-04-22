# ONNX Runtime Configuration

ONNX Runtime's execution providers enable to perform inferences with hardware acceleration on GPUs or NPUs, resulting in a massive performance boost.

**Primarily refer to the [ort documentation](https://ort.pyke.io/perf/execution-providers) for details about the execution providers and their particular requirements.**

## Related Cargo Features in `gline-rs`

Some features of `ort` are mirrored to allow enabling the necessary execution providers. For example `cuda`, `coreml` etc. See the main `Readme.md` for the complete list of available features.

The `load-dynamic` feature allows for dynamic loading of the ONNX runtime. This is useful if your platform does not support static linking, or if it does not work for some reason.

## Hints for Specific Configurations

### CUDA on Windows

Some configurations should work out of the box with static linkage, but some others might not be available or have problems due to conflicting libraries for example. In such case the best bet is probably to manually install appropriate libraries and enable dynamic linking (see above).

For CUDA on Windows the following packages are needed: 

* ONNX Runtime (v1.20)
* CUDA Toolkit (v12.x)
* CUDNN Toolkit (9.x)

Then it is **mandatory** to set some environments variables for the DLLs to load properly, pointing to the actual locations of the three libraries. For example:

```bat
SET ORT_DYLIB_PATH=C:\Program Files\onnxruntime-win-x64-gpu-1.20.1\lib\onnxruntime.dll
SET PATH=%PATH%;"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
SET PATH=%PATH%;"C:\Program Files\NVIDIA\CUDNN\v9.6\bin\12.6"

```

Then, enabling the features `cuda` and `load-dynamic` should work as expected:

```dos
> cargo run --example benchmark-gpu --features=cuda,load-dynamic
```