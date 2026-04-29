# 🔌 API Changes v0.10.0

## Parakeet streaming via `transcribeStream`

PR: [#TBD]

The `transcribeStream` duplex op now accepts parakeet model IDs. The parakeet
plugin handler routes the live PCM stream into the new
`TranscriptionParakeet.runStreaming` entrypoint, which drives a Silero-VAD-
based streaming engine inside the parakeet addon. Whisper streaming is
unchanged.

```typescript
import { transcribeStream } from "@qvac/sdk";

// Duplex session: stream audio in, stream transcripts out.
const session = await transcribeStream({ modelId });
for await (const chunk of micPcm) session.write(chunk);
session.end();

for await (const seg of session) {
  if (seg.isPartial) replaceRunningTail(seg.text);
  else commit(seg.text);
}
session.destroy();
```

---

## `vadModelSrc` and `vad_params` on the parakeet config

PR: [#TBD]

`parakeetConfigSchema` accepts an optional `vadModelSrc` (the Silero VAD model
source) and an optional `vad_params` object. Setting `vadModelSrc` activates
the streaming path; leaving it unset keeps the legacy one-shot `transcribe()`
flow untouched.

```typescript
import { loadModel, VAD_SILERO_5_1_2 } from "@qvac/sdk";

await loadModel({
  modelType: "parakeet-transcription",
  config: {
    parakeetEncoderSrc: ENCODER,
    parakeetDecoderSrc: DECODER,
    parakeetPreprocessorSrc: PREPROCESSOR,
    parakeetVocabSrc: VOCAB,
    vadModelSrc: VAD_SILERO_5_1_2,        // activates streaming
    vad_params: {                         // all fields optional
      threshold: 0.5,
      min_silence_duration_ms: 500,
      min_speech_duration_ms: 250,
      max_speech_duration_s: 30,
      speech_pad_ms: 30,
      samples_overlap: 0.1,
      partial_decode_interval_ms: 1500,   // 0 disables partials
    },
  },
});
```

The addon falls back to whisper-equivalent defaults for any field not set on
`vad_params`. Setting `partial_decode_interval_ms: 0` opts out of mid-segment
partial decoding (final-only stream).

---

## `TranscribeStreamSession` iterator yields `{text, isPartial?}`

PR: [#TBD]

`transcribeStreamResponseSchema` gains an optional `isPartial: boolean`.
Parakeet emits live mid-segment partials (`isPartial: true`) that consumers
should replace until the next final commit (`isPartial: false`). Whisper
continues to emit final-only frames; for engines that don't set the flag the
field is simply absent and existing consumers are unaffected.

```typescript
// Before
for await (const { text } of session) commit(text);

// After (parakeet partials supported)
for await (const { text, isPartial } of session) {
  if (isPartial) replaceTail(text);
  else commit(text);
}
```
