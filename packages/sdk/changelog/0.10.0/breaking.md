# 💥 Breaking Changes v0.10.0

No breaking changes.

All new fields on `parakeetConfigSchema` (`vadModelSrc`, `vad_params`) and on
`transcribeStreamResponseSchema` (`isPartial`) are optional. Existing
consumers of `transcribe()`, the EOU streaming path, and the whisper
`transcribeStream` flow are unaffected.
