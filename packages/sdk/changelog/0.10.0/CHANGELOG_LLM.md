# QVAC SDK v0.10.0 Release Notes

📦 **NPM:** https://www.npmjs.com/package/@qvac/sdk/v/0.10.0

This release brings the `transcribeStream` duplex op to Parakeet, reaching parity with Whisper for live transcription. Callers that pass a `vadModelSrc` get VAD-driven simulated streaming powered by Silero v5 inside the Parakeet addon, with optional mid-segment partial decoding for live-dictation UX. Whisper streaming is unchanged.

---

## ✨ Features

### Parakeet streaming via `transcribeStream`

The `transcribeStream` duplex op now accepts parakeet model IDs. The parakeet plugin handler routes the live PCM stream into `TranscriptionParakeet.runStreaming`, which drives the new native `StreamingProcessor` (Silero VAD + offline Parakeet recognizer). Final segments are emitted as `{text, isPartial: false}`; mid-segment partials (when enabled) as `{text, isPartial: true}`.

```typescript
import { transcribeStream } from "@qvac/sdk";

const session = await transcribeStream({ modelId });
for await (const chunk of micPcm) session.write(chunk);
session.end();
for await (const seg of session) {
  if (seg.isPartial) replaceRunningTail(seg.text);
  else commit(seg.text);
}
```

### `vadModelSrc` and `vad_params` on the parakeet config

```typescript
loadModel({
  modelType: "parakeet-transcription",
  config: {
    parakeetEncoderSrc: ENCODER,
    parakeetDecoderSrc: DECODER,
    parakeetPreprocessorSrc: PREPROCESSOR,
    parakeetVocabSrc: VOCAB,
    vadModelSrc: VAD_SILERO_5_1_2,     // activates streaming
    vad_params: {
      threshold: 0.5,
      min_silence_duration_ms: 500,
      partial_decode_interval_ms: 1500, // mid-segment partials
    },
  },
});
```

All `vad_params` fields are optional; the addon falls back to whisper-equivalent defaults (threshold 0.5, min silence 500 ms, min speech 250 ms, max speech 30 s, pad 30 ms, overlap 0.1, partial interval 0 = disabled).

## 🔌 API Changes

### `TranscribeStreamSession` iterator yields `{text, isPartial}`

The duplex iterator was previously typed as `AsyncIterable<{text}>`. It is now `AsyncIterable<{text, isPartial?}>`. Whisper engines do not emit `isPartial`; existing consumers that only read `text` keep working unchanged.

```typescript
// Before
for await (const { text } of session) commit(text);

// After (parakeet partials supported)
for await (const { text, isPartial } of session) {
  if (isPartial) replaceTail(text);
  else commit(text);
}
```

## 📦 Models

No new model constants were added in this release. Parakeet streaming consumes the existing `VAD_SILERO_5_1_2` constant (`silero_vad.onnx`, ~2 MB, MIT, [snakers4/silero-vad](https://github.com/snakers4/silero-vad)). Whisper streaming is unaffected and continues to use its built-in `whisper_vad_context`.

## ⬆️ Dependencies

- `@qvac/transcription-parakeet` peer range: `^0.3.1` → `^0.4.0`. The new streaming entrypoints (`runStreaming`, `startStreaming`, `appendStreamingAudio`, `endStreaming`) ship in 0.4.0.
