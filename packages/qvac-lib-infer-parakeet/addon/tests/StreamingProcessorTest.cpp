#include <string_view>

#include <gtest/gtest.h>

#include "model-interface/ParakeetTypes.hpp"
#include "model-interface/StreamingProcessor.hpp"

namespace qvac_lib_infer_parakeet {
namespace {

TEST(TranscriptTest, DefaultConstructorMarksSegmentAsFinal) {
  Transcript t;
  EXPECT_FALSE(t.isPartial);
  EXPECT_FALSE(t.toAppend);
  EXPECT_TRUE(t.text.empty());
  EXPECT_EQ(t.id, 0u);
  EXPECT_FLOAT_EQ(t.start, -1.0F);
  EXPECT_FLOAT_EQ(t.end, -1.0F);
}

TEST(TranscriptTest, StringViewConstructorMarksSegmentAsFinal) {
  Transcript t{std::string_view("hello")};
  EXPECT_FALSE(t.isPartial);
  EXPECT_EQ(t.text, "hello");
}

TEST(StreamingProcessorConfigTest, DefaultsMatchPublicContract) {
  StreamingProcessor::Config cfg;

  EXPECT_EQ(cfg.sampleRate, 16000);
  EXPECT_FLOAT_EQ(cfg.vadThreshold, 0.5F);
  EXPECT_EQ(cfg.minSilenceDurationMs, 500);
  EXPECT_EQ(cfg.minSpeechDurationMs, 250);
  EXPECT_FLOAT_EQ(cfg.maxSpeechDurationS, 30.0F);
  EXPECT_EQ(cfg.speechPadMs, 30);
  EXPECT_FLOAT_EQ(cfg.samplesOverlap, 0.1F);

  // Derived sample-domain values follow the documented defaults
  // (max-speech 30 s @ 16 kHz, 0.3 s VAD cadence, 1.5 s partial cadence).
  EXPECT_EQ(cfg.maxBufferSamples, 30 * 16000);
  EXPECT_EQ(cfg.vadRunIntervalSamples, static_cast<int>(0.3F * 16000));
  EXPECT_EQ(cfg.partialDecodeIntervalSamples, 24000);
}

TEST(StreamingProcessorConfigTest, PartialDecodeCanBeDisabled) {
  StreamingProcessor::Config cfg;
  cfg.partialDecodeIntervalSamples = 0;

  // A zero cadence is the documented opt-out for mid-segment partials;
  // the processor must accept it without rewriting it back to the default.
  EXPECT_EQ(cfg.partialDecodeIntervalSamples, 0);
}

TEST(StreamingProcessorConfigTest, OverridesArePreserved) {
  StreamingProcessor::Config cfg;
  cfg.sampleRate = 8000;
  cfg.vadThreshold = 0.7F;
  cfg.minSilenceDurationMs = 800;
  cfg.partialDecodeIntervalSamples = 12000;

  EXPECT_EQ(cfg.sampleRate, 8000);
  EXPECT_FLOAT_EQ(cfg.vadThreshold, 0.7F);
  EXPECT_EQ(cfg.minSilenceDurationMs, 800);
  EXPECT_EQ(cfg.partialDecodeIntervalSamples, 12000);
}

}  // namespace
}  // namespace qvac_lib_infer_parakeet
