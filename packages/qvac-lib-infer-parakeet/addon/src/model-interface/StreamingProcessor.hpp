#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "SileroVad.hpp"
#include "qvac-lib-inference-addon-cpp/queue/OutputQueue.hpp"

namespace qvac_lib_infer_parakeet {

class ParakeetModel;

// VAD-driven simulated-streaming processor for the Parakeet addon.
// Mirrors qvac_lib_inference_addon_whisper::StreamingProcessor one-to-one
// but uses SileroVad (ONNX session) in place of whisper_vad_context and
// routes transcripts through ParakeetModel::process + takeOutput.
class StreamingProcessor {
public:
  struct Config {
    std::uint64_t jobId = 0;
    static constexpr int kDefaultSampleRate = 16000;
    static constexpr float kDefaultMaxSpeechDurationS = 30.0F;
    static constexpr float kVadRunIntervalS = 0.3F;
    // Default cadence for in-segment partial decoding: re-run the recognizer
    // every 1.5 s of new audio inside a still-open VAD segment (24 000
    // samples @ 16 kHz). 0 disables partial decoding entirely (final-only).
    static constexpr float kDefaultPartialDecodeIntervalS = 1.5F;

    int sampleRate = kDefaultSampleRate;
    std::string vadModelPath;
    float vadThreshold = 0.5F;
    int minSilenceDurationMs = 500;
    int minSpeechDurationMs = 250;
    float maxSpeechDurationS = kDefaultMaxSpeechDurationS;
    int speechPadMs = 30;
    float samplesOverlap = 0.1F;
    int maxBufferSamples =
        static_cast<int>(kDefaultMaxSpeechDurationS) * kDefaultSampleRate;
    int vadRunIntervalSamples =
        static_cast<int>(kVadRunIntervalS * kDefaultSampleRate);
    // Minimum new audio (samples) accumulated since the last partial decode
    // before the in-progress segment is re-decoded and emitted as a partial.
    // Set to 0 to disable mid-segment partials (legacy behavior — final-only).
    int partialDecodeIntervalSamples =
        static_cast<int>(kDefaultPartialDecodeIntervalS * kDefaultSampleRate);
  };

  StreamingProcessor(
      ParakeetModel& model,
      std::shared_ptr<qvac_lib_inference_addon_cpp::OutputQueue> outputQueue,
      Config config);

  ~StreamingProcessor();

  StreamingProcessor(const StreamingProcessor&) = delete;
  StreamingProcessor& operator=(const StreamingProcessor&) = delete;
  StreamingProcessor(StreamingProcessor&&) = delete;
  StreamingProcessor& operator=(StreamingProcessor&&) = delete;

  void appendAudio(std::vector<float>&& samples);
  void end();
  void cancel();

private:
  void processLoop();
  void processAudioRange(int startSample, int endSample, bool isPartial);

  ParakeetModel& model_;
  std::shared_ptr<qvac_lib_inference_addon_cpp::OutputQueue> outputQueue_;
  Config config_;

  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::vector<float> pendingAudio_;
  std::vector<float> processBuffer_;
  bool ended_ = false;
  bool cancelled_ = false;
  bool hasError_ = false;

  std::unique_ptr<SileroVad> vad_;
  int bufferSizeAtLastVadRun_ = 0;
  // Buffer size (in samples, on processBuffer_) at the moment we last
  // emitted a partial result for the currently-open VAD segment. Reset to 0
  // every time the buffer is trimmed after a final segment commit, so the
  // partial cadence restarts from scratch on each fresh utterance.
  int bufferSizeAtLastPartialDecode_ = 0;

  std::thread thread_;
};

} // namespace qvac_lib_infer_parakeet
