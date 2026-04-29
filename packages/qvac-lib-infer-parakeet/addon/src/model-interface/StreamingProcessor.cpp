#include "StreamingProcessor.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "parakeet/ParakeetModel.hpp"
#include "qvac-lib-inference-addon-cpp/Logger.hpp"

namespace qvac_lib_infer_parakeet {

StreamingProcessor::StreamingProcessor(
    ParakeetModel& model,
    std::shared_ptr<qvac_lib_inference_addon_cpp::OutputQueue> outputQueue,
    Config config)
    : model_(model), outputQueue_(std::move(outputQueue)),
      config_(std::move(config)) {

  if (config_.vadModelPath.empty()) {
    throw std::runtime_error(
        "StreamingProcessor: vadModelPath is required");
  }

  vad_ = std::make_unique<SileroVad>(config_.vadModelPath, config_.sampleRate);

  QLOG(
      qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
      "StreamingProcessor: VAD initialized from " + config_.vadModelPath);

  thread_ = std::thread([this]() { processLoop(); });
}

StreamingProcessor::~StreamingProcessor() {
  {
    std::lock_guard lock(mtx_);
    ended_ = true;
  }
  cv_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
  vad_.reset();
}

void StreamingProcessor::appendAudio(std::vector<float>&& samples) {
  {
    std::lock_guard lock(mtx_);
    if (ended_) {
      return;
    }
    if (pendingAudio_.empty()) {
      pendingAudio_ = std::move(samples);
    } else {
      pendingAudio_.insert(pendingAudio_.end(), samples.begin(), samples.end());
    }
    if (static_cast<int>(pendingAudio_.size()) > config_.maxBufferSamples) {
      const int excess =
          static_cast<int>(pendingAudio_.size()) - config_.maxBufferSamples;
      pendingAudio_.erase(
          pendingAudio_.begin(), pendingAudio_.begin() + excess);
      QLOG(
          qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
          "StreamingProcessor: dropped " + std::to_string(excess) +
              " samples from pendingAudio_ (backpressure)");
    }
  }
  cv_.notify_one();
}

void StreamingProcessor::end() {
  {
    std::lock_guard lock(mtx_);
    ended_ = true;
  }
  cv_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void StreamingProcessor::cancel() {
  model_.cancel();
  {
    std::lock_guard lock(mtx_);
    cancelled_ = true;
    ended_ = true;
  }
  cv_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void StreamingProcessor::processAudioRange(
    int startSample, int endSample, bool isPartial) {
  const int len = endSample - startSample;
  if (len <= 0) {
    return;
  }

  QLOG(
      qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
      std::string("StreamingProcessor: processing ") + std::to_string(len) +
          " samples (" +
          std::to_string(
              static_cast<double>(len) /
              static_cast<double>(config_.sampleRate)) +
          "s)" + (isPartial ? " [partial]" : " [final]"));

  ParakeetModel::Input segment(
      processBuffer_.begin() + startSample,
      processBuffer_.begin() + endSample);

  try {
    model_.process(segment);
    auto transcripts = model_.takeOutput();
    if (!transcripts.empty()) {
      // Tag every transcript in this batch so the consumer can decide
      // whether to replace the running partial in the UI or commit it.
      for (auto& t : transcripts) {
        t.isPartial = isPartial;
      }
      outputQueue_->queueResult(std::any(std::move(transcripts)));
    }
  } catch (const std::exception& e) {
    hasError_ = true;
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::ERROR,
        std::string("StreamingProcessor: processing error: ") + e.what());
  }
}

void StreamingProcessor::processLoop() {
  QLOG(
      qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
      "StreamingProcessor: thread started (VAD-based segmentation)");

  model_.prepareForStreaming();

  SileroVad::Params vadParams;
  vadParams.threshold = config_.vadThreshold;
  vadParams.minSpeechDurationMs = config_.minSpeechDurationMs;
  vadParams.minSilenceDurationMs = config_.minSilenceDurationMs;
  vadParams.maxSpeechDurationS = config_.maxSpeechDurationS;
  vadParams.speechPadMs = config_.speechPadMs;
  vadParams.samplesOverlap = config_.samplesOverlap;

  while (true) {
    bool done = false;
    bool wasCancelled = false;

    {
      std::unique_lock lock(mtx_);
      cv_.wait(lock, [this]() {
        return ended_ || !pendingAudio_.empty();
      });

      processBuffer_.insert(
          processBuffer_.end(), pendingAudio_.begin(), pendingAudio_.end());
      pendingAudio_.clear();

      done = ended_;
      wasCancelled = cancelled_;
    }

    if (wasCancelled) {
      break;
    }

    const int bufferSize = static_cast<int>(processBuffer_.size());

    const bool shouldRunVad =
        (bufferSize - bufferSizeAtLastVadRun_) >=
            config_.vadRunIntervalSamples ||
        done;

    if (shouldRunVad && bufferSize > 0) {
      bufferSizeAtLastVadRun_ = bufferSize;

      // Wrap VAD in try/catch so a Silero inference failure (wrong input
      // shape, corrupt weights, etc) doesn't silently kill the thread and
      // leave the client's response iterator hanging forever.
      std::vector<SileroVad::Segment> segments;
      try {
        segments =
            vad_->getSegments(processBuffer_.data(), bufferSize, vadParams);
      } catch (const std::exception& e) {
        hasError_ = true;
        QLOG(
            qvac_lib_inference_addon_cpp::logger::Priority::ERROR,
            std::string("StreamingProcessor: VAD error: ") + e.what());
        break;
      }
      const int nSeg = static_cast<int>(segments.size());

      if (nSeg > 0) {
        const float totalDurationS =
            static_cast<float>(bufferSize) /
            static_cast<float>(config_.sampleRate);

        constexpr float CS_TO_SEC = 0.01F;

        int lastComplete = -1;
        for (int i = 0; i < nSeg; ++i) {
          const float t1S = static_cast<float>(segments[i].t1Cs) * CS_TO_SEC;
          const float marginS =
              static_cast<float>(config_.speechPadMs) / 1000.0F;
          if (t1S + marginS < totalDurationS) {
            lastComplete = i;
          }
        }

        if (done) {
          lastComplete = nSeg - 1;
        }

        if (lastComplete >= 0) {
          QLOG(
              qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
              "StreamingProcessor: VAD found " + std::to_string(nSeg) +
                  " segment(s), " + std::to_string(lastComplete + 1) +
                  " complete, totalDuration=" +
                  std::to_string(totalDurationS) + "s");

          for (int i = 0; i <= lastComplete; ++i) {
            const float t0S =
                static_cast<float>(segments[i].t0Cs) * CS_TO_SEC;
            const float t1S =
                static_cast<float>(segments[i].t1Cs) * CS_TO_SEC;
            const int startSample = std::max(
                0,
                static_cast<int>(
                    t0S * static_cast<float>(config_.sampleRate)));
            const int endSample = std::min(
                static_cast<int>(
                    t1S * static_cast<float>(config_.sampleRate)),
                bufferSize);
            if (endSample > startSample) {
              processAudioRange(startSample, endSample, /*isPartial=*/false);
            }
          }

          const float lastT1S =
              static_cast<float>(segments[lastComplete].t1Cs) * CS_TO_SEC;
          const int trimPoint = std::min(
              static_cast<int>(
                  lastT1S * static_cast<float>(config_.sampleRate)),
              bufferSize);
          processBuffer_.erase(
              processBuffer_.begin(), processBuffer_.begin() + trimPoint);
          bufferSizeAtLastVadRun_ = 0;
          // Buffer was trimmed up to the end of the last final segment; the
          // partial cadence restarts on the fresh utterance that follows.
          bufferSizeAtLastPartialDecode_ = 0;
        }

        // Mid-segment partial decode: when the most recent VAD segment is
        // still open (its end + speechPad has not yet cleared the live
        // buffer) and enough new audio has accumulated since our last
        // partial pass, run Parakeet on the in-progress range and emit it
        // tagged isPartial=true. Skipped on the terminating pass (done) so
        // we don't compete with the natural end-of-stream final flush, and
        // skipped when complete segments were just trimmed this pass — the
        // remaining buffer is then a fresh utterance with stale VAD coords,
        // which we'll re-segment cleanly on the next loop iteration.
        if (!done && config_.partialDecodeIntervalSamples > 0 &&
            !segments.empty() && lastComplete < 0) {
          const int currentBufferSize =
              static_cast<int>(processBuffer_.size());
          const SileroVad::Segment& lastSeg = segments.back();
          const float lastT0S =
              static_cast<float>(lastSeg.t0Cs) * CS_TO_SEC;
          const float lastT1S =
              static_cast<float>(lastSeg.t1Cs) * CS_TO_SEC;
          const float marginS =
              static_cast<float>(config_.speechPadMs) / 1000.0F;
          const float liveDurationS =
              static_cast<float>(currentBufferSize) /
              static_cast<float>(config_.sampleRate);
          const bool segmentStillOpen = lastT1S + marginS >= liveDurationS;

          if (segmentStillOpen) {
            const int newAudioSinceLast =
                currentBufferSize - bufferSizeAtLastPartialDecode_;
            if (newAudioSinceLast >= config_.partialDecodeIntervalSamples) {
              const int partialStart = std::max(
                  0,
                  static_cast<int>(
                      lastT0S *
                      static_cast<float>(config_.sampleRate)));
              const int partialEnd = currentBufferSize;
              if (partialEnd > partialStart) {
                processAudioRange(
                    partialStart, partialEnd, /*isPartial=*/true);
                bufferSizeAtLastPartialDecode_ = currentBufferSize;
              }
            }
          }
        }
      }

      if (static_cast<int>(processBuffer_.size()) >=
          config_.maxBufferSamples) {
        QLOG(
            qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
            "StreamingProcessor: buffer overflow, force-processing " +
                std::to_string(processBuffer_.size()) + " samples");
        processAudioRange(
            0, static_cast<int>(processBuffer_.size()), /*isPartial=*/false);
        processBuffer_.clear();
        bufferSizeAtLastVadRun_ = 0;
        bufferSizeAtLastPartialDecode_ = 0;
      }
    }

    if (done) {
      break;
    }
  }

  {
    std::lock_guard lock(mtx_);
    if (cancelled_) {
      outputQueue_->queueException(std::runtime_error("Job cancelled"));
      return;
    }
  }

  if (!processBuffer_.empty()) {
    processAudioRange(
        0, static_cast<int>(processBuffer_.size()), /*isPartial=*/false);
    processBuffer_.clear();
  }

  if (hasError_) {
    outputQueue_->queueException(std::runtime_error(
        "StreamingProcessor: one or more segments failed during processing"));
  } else {
    outputQueue_->queueJobEnded();
  }
}

} // namespace qvac_lib_infer_parakeet
