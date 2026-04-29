#include "SileroVad.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace qvac_lib_infer_parakeet {

namespace {

// Silero v5 uses a fixed 512-sample window at 16 kHz (32 ms).
constexpr int kSileroWindowSamples16k = 512;
// LSTM hidden state tensor shape: [2, 1, 128]  (layers*directions, batch, hidden).
constexpr int kSileroStateSize = 2 * 1 * 128;

} // namespace

SileroVad::SileroVad(const std::string& modelPath, int sampleRate)
    : sampleRate_(sampleRate) {
  if (sampleRate_ != 16000 && sampleRate_ != 8000) {
    throw std::runtime_error(
        "SileroVad: unsupported sample rate " + std::to_string(sampleRate_) +
        " (only 8000 or 16000 are allowed)");
  }

  windowSamples_ = (sampleRate_ == 16000) ? kSileroWindowSamples16k
                                          : kSileroWindowSamples16k / 2;

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SileroVad");

  options_ = std::make_unique<Ort::SessionOptions>();
  options_->SetInterOpNumThreads(1);
  options_->SetIntraOpNumThreads(1);
  options_->SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
  // Ort::Session on Windows expects a wide-char path.
  std::wstring widePath(modelPath.begin(), modelPath.end());
  session_ = std::make_unique<Ort::Session>(
      *env_, widePath.c_str(), *options_);
#else
  session_ = std::make_unique<Ort::Session>(
      *env_, modelPath.c_str(), *options_);
#endif

  memInfo_ = std::make_unique<Ort::MemoryInfo>(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t numInputs = session_->GetInputCount();
  inputNames_.reserve(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator);
    inputNames_.emplace_back(name.get());
  }
  const size_t numOutputs = session_->GetOutputCount();
  outputNames_.reserve(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator);
    outputNames_.emplace_back(name.get());
  }

  state_.assign(kSileroStateSize, 0.0F);
}

SileroVad::~SileroVad() = default;

void SileroVad::reset() {
  std::fill(state_.begin(), state_.end(), 0.0F);
}

float SileroVad::runFrame(const float* frame, int frameSamples) {
  // Silero v5 inputs: { input [1, window], state [2,1,128], sr [1] }
  // Outputs:          { output [1, 1], state_out [2,1,128] }
  std::array<int64_t, 2> inputShape = {1, static_cast<int64_t>(frameSamples)};
  std::array<int64_t, 3> stateShape = {2, 1, 128};
  std::array<int64_t, 1> srShape = {1};
  std::array<int64_t, 1> srData = {static_cast<int64_t>(sampleRate_)};

  std::vector<Ort::Value> inputs;
  inputs.reserve(3);
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      *memInfo_, const_cast<float*>(frame), frameSamples, inputShape.data(),
      inputShape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      *memInfo_, state_.data(), state_.size(), stateShape.data(),
      stateShape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
      *memInfo_, srData.data(), srData.size(), srShape.data(),
      srShape.size()));

  std::vector<const char*> inputNamePtrs;
  inputNamePtrs.reserve(inputNames_.size());
  for (const auto& n : inputNames_) inputNamePtrs.push_back(n.c_str());
  std::vector<const char*> outputNamePtrs;
  outputNamePtrs.reserve(outputNames_.size());
  for (const auto& n : outputNames_) outputNamePtrs.push_back(n.c_str());

  auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, inputNamePtrs.data(), inputs.data(),
      inputs.size(), outputNamePtrs.data(), outputNamePtrs.size());

  if (outputs.size() < 2) {
    throw std::runtime_error("SileroVad: unexpected output count");
  }

  const float prob = *outputs[0].GetTensorData<float>();

  // Persist new LSTM state for the next frame.
  const auto* newState = outputs[1].GetTensorData<float>();
  std::memcpy(state_.data(), newState, state_.size() * sizeof(float));

  return prob;
}

std::vector<SileroVad::Segment> SileroVad::getSegments(
    const float* samples, int numSamples, const Params& params) {
  // Reset LSTM state so the probability track only reflects the current
  // buffer — StreamingProcessor hands us the full pendingAudio buffer each
  // time and we must be deterministic across calls.
  reset();

  const int window = windowSamples_;
  if (window <= 0 || numSamples < window) {
    return {};
  }

  const int numWindows = numSamples / window;
  std::vector<float> probs;
  probs.reserve(numWindows);
  for (int i = 0; i < numWindows; ++i) {
    probs.push_back(runFrame(samples + (i * window), window));
  }

  // Threshold + hysteresis segmentation (mirrors whisper.cpp's VAD logic).
  const float threshold = params.threshold;
  const float negThreshold = std::max(0.0F, threshold - 0.15F);
  const int minSpeechFrames = std::max(
      1, static_cast<int>(std::round(
             static_cast<float>(params.minSpeechDurationMs) /
             1000.0F * static_cast<float>(sampleRate_) /
             static_cast<float>(window))));
  const int minSilenceFrames = std::max(
      1, static_cast<int>(std::round(
             static_cast<float>(params.minSilenceDurationMs) /
             1000.0F * static_cast<float>(sampleRate_) /
             static_cast<float>(window))));
  const int maxSpeechFrames = std::max(
      minSpeechFrames + 1,
      static_cast<int>(std::round(
          params.maxSpeechDurationS * static_cast<float>(sampleRate_) /
          static_cast<float>(window))));
  const int padFrames = std::max(
      0, static_cast<int>(std::round(
             static_cast<float>(params.speechPadMs) / 1000.0F *
             static_cast<float>(sampleRate_) /
             static_cast<float>(window))));

  std::vector<Segment> segments;
  bool inSpeech = false;
  int speechStartFrame = 0;
  int silenceStartFrame = -1;

  auto framesToCs = [&](int frames) {
    // centiseconds = frames * window * 100 / sampleRate
    return (frames * window * 100) / sampleRate_;
  };

  for (int i = 0; i < numWindows; ++i) {
    const float p = probs[i];
    if (!inSpeech) {
      if (p >= threshold) {
        inSpeech = true;
        speechStartFrame = std::max(0, i - padFrames);
        silenceStartFrame = -1;
      }
      continue;
    }

    // Currently inside a speech segment.
    if (p < negThreshold) {
      if (silenceStartFrame < 0) {
        silenceStartFrame = i;
      }
      const int silenceLen = i - silenceStartFrame + 1;
      if (silenceLen >= minSilenceFrames) {
        const int speechEndFrame =
            std::min(numWindows, silenceStartFrame + padFrames);
        const int speechLen = speechEndFrame - speechStartFrame;
        if (speechLen >= minSpeechFrames) {
          segments.push_back(
              {framesToCs(speechStartFrame), framesToCs(speechEndFrame)});
        }
        inSpeech = false;
        silenceStartFrame = -1;
      }
    } else {
      silenceStartFrame = -1;
    }

    // Hard cap on maximum speech length.
    if (inSpeech && (i - speechStartFrame) >= maxSpeechFrames) {
      const int speechEndFrame = i + 1;
      segments.push_back(
          {framesToCs(speechStartFrame), framesToCs(speechEndFrame)});
      inSpeech = false;
      silenceStartFrame = -1;
    }
  }

  // Trailing speech that never closed — leave open for the next invocation
  // unless the caller is at end of stream (handled at the
  // StreamingProcessor level via the "force-process final buffer" path).
  return segments;
}

} // namespace qvac_lib_infer_parakeet
