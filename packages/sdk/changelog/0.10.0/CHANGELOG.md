# Changelog v0.10.0

Release Date: 2026-04-26

## ✨ Features

- Parakeet now supports VAD-driven simulated streaming via the existing `transcribeStream` duplex op. Previously this op was Whisper-only; the parakeet plugin gains a `transcribeStream` handler that routes live PCM into the new `runStreaming` API of `@qvac/transcription-parakeet`.
- `parakeetConfigSchema` accepts a `vadModelSrc` (Silero VAD model source) and a `vad_params` object (threshold, min/max speech & silence durations, speech pad, samples overlap, partial decode cadence). All fields are optional; setting `vadModelSrc` activates streaming, leaving it unset keeps the legacy `transcribe()` flow.
- `transcribeStreamResponseSchema` gains an optional `isPartial` boolean. Parakeet emits live mid-segment partials (`isPartial: true`) that consumers should replace until the next final segment commit (`isPartial: false`). Whisper continues to emit final-only frames.
- `transcribe-config` schema exposes `partial_decode_interval_ms` so callers can tune the partial cadence (0 disables partials).

## 🔌 API Changes

- Client `TranscribeStreamSession` iterator now yields `{text, isPartial}` instead of `{text}`. The `isPartial` field is optional in TypeScript and absent for engines that don't support partials, so existing whisper consumers are unaffected.

## 📦 Models

- No new model constants. Parakeet streaming consumes the existing `VAD_SILERO_5_1_2` constant (`silero_vad.onnx`, ~2 MB, MIT, [snakers4/silero-vad](https://github.com/snakers4/silero-vad)). Required only when using parakeet streaming.

## ⬆️ Dependencies

- `@qvac/transcription-parakeet` peer range bumped to `^0.4.0` (was `^0.3.1`) — the new streaming entrypoints live there.
