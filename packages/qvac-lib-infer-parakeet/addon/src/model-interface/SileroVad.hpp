#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Ort {
class Env;
class Session;
class SessionOptions;
class MemoryInfo;
} // namespace Ort

namespace qvac_lib_infer_parakeet {

// Standalone Silero v5 VAD running on ONNX Runtime.
//
// Parakeet's addon historically had no VAD — whisper.cpp bakes Silero into
// its own C++ library, but parakeet links onnxruntime via @qvac/onnx.
// This class opens a dedicated Ort::Session for silero_vad.onnx and exposes
// a segmentation API modelled after whisper_vad_segments_from_samples so
// StreamingProcessor can stay symmetric with the whisper counterpart.
//
// Silero v5 operates on 16 kHz mono s16le audio. It consumes fixed-size
// windows (512 samples, 32 ms) and returns a speech probability per window.
// Hidden state (LSTM h/c concatenated into a single [2, 1, 128] tensor) is
// carried across calls by the caller to keep inference stateful.
class SileroVad {
public:
  struct Segment {
    // Boundaries in centiseconds (same unit whisper uses) so the integration
    // point in StreamingProcessor remains a drop-in.
    int t0Cs;
    int t1Cs;
  };

  struct Params {
    float threshold = 0.5F;
    int minSpeechDurationMs = 250;
    int minSilenceDurationMs = 500;
    float maxSpeechDurationS = 30.0F;
    int speechPadMs = 30;
    float samplesOverlap = 0.1F;
  };

  SileroVad(const std::string& modelPath, int sampleRate);
  ~SileroVad();

  SileroVad(const SileroVad&) = delete;
  SileroVad& operator=(const SileroVad&) = delete;
  SileroVad(SileroVad&&) = delete;
  SileroVad& operator=(SileroVad&&) = delete;

  // Run the VAD over `samples` (float mono PCM at `sampleRate_`) and return
  // the speech segments detected. The implementation re-runs the model over
  // the full buffer on every call; StreamingProcessor only invokes us once
  // new audio has accumulated, matching whisper.cpp's semantics.
  std::vector<Segment>
  getSegments(const float* samples, int numSamples, const Params& params);

  // Reset the internal LSTM state (h/c).
  void reset();

  int sampleRate() const { return sampleRate_; }

private:
  float runFrame(const float* frame, int frameSamples);

  int sampleRate_;
  int windowSamples_ = 0;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> options_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memInfo_;

  // LSTM hidden state: single tensor [2, 1, 128] (Silero v5).
  std::vector<float> state_;

  // Cached IO names owned by the session allocator.
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
};

} // namespace qvac_lib_infer_parakeet
