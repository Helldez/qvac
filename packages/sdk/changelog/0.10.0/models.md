# 📦 Model Changes v0.10.0

No new model constants were added in this release.

## Newly consumed by parakeet streaming

```
VAD_SILERO_5_1_2
```

The constant already exists in the SDK model registry. Parakeet streaming
loads `silero_vad.onnx` (~2 MB, MIT, [snakers4/silero-vad](https://github.com/snakers4/silero-vad))
through this constant when `vadModelSrc` is set on the parakeet config.
Whisper streaming is unaffected and continues to use its built-in
`whisper_vad_context`.
