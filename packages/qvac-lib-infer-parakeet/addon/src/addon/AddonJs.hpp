#pragma once

#include <any>
#include <memory>
#include <string>
#include <vector>

#include <js.h>

#include <qvac-lib-inference-addon-cpp/JsInterface.hpp>
#include <qvac-lib-inference-addon-cpp/JsUtils.hpp>
#include <qvac-lib-inference-addon-cpp/ModelInterfaces.hpp>
#include <qvac-lib-inference-addon-cpp/addon/AddonJs.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/OutputHandler.hpp>
#include <qvac-lib-inference-addon-cpp/queue/OutputCallbackJs.hpp>

#include "model-interface/ParakeetTypes.hpp"
#include "model-interface/StreamingProcessor.hpp"
#include "model-interface/parakeet/ParakeetModel.hpp"
#include "js-interface/JSAdapter.hpp"

#include <map>
#include <mutex>

namespace qvac_lib_infer_parakeet {

// Active streaming sessions indexed by addon instance pointer. Lives in the
// header because only the JS-facing handlers touch it, and keeping it inline
// avoids adding a new translation unit to the build.
inline std::mutex g_streamingMtx;
inline std::map<
    qvac_lib_inference_addon_cpp::AddonJs*,
    std::unique_ptr<StreamingProcessor>>
    g_streamingSessions;

namespace js = qvac_lib_inference_addon_cpp::js;

inline ParakeetConfig createParakeetConfig(
    js_env_t* env, const js::Object& configurationParams) {
  JSAdapter adapter;
  return adapter.loadFromJSObject(configurationParams, env);
}

struct JsParakeetOutputHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
          std::vector<Transcript>> {
  JsParakeetOutputHandler()
      : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
            std::vector<Transcript>>(
            [this](const std::vector<Transcript>& output) -> js_value_t* {
              auto jsOutput = js::Array::create(this->env_);
              for (size_t i = 0; i < output.size(); ++i) {
                auto jsTranscript = js::Object::create(this->env_);
                jsTranscript.setProperty(
                    this->env_,
                    "text",
                    js::String::create(this->env_, output[i].text));
                jsTranscript.setProperty(
                    this->env_,
                    "toAppend",
                    js::Boolean::create(this->env_, output[i].toAppend));
                jsTranscript.setProperty(
                    this->env_,
                    "start",
                    js::Number::create(this->env_, output[i].start));
                jsTranscript.setProperty(
                    this->env_,
                    "end",
                    js::Number::create(this->env_, output[i].end));
                jsTranscript.setProperty(
                    this->env_,
                    "id",
                    js::Number::create(
                        this->env_, static_cast<uint64_t>(output[i].id)));
                jsOutput.set(this->env_, i, jsTranscript);
              }
              return jsOutput;
            }) {}
};

inline js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  auto configurationParams = args.getJsObject(1, "configurationParams");

  unique_ptr<model::IModel> model =
      make_unique<ParakeetModel>(createParakeetConfig(env, configurationParams));

  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outputHandlers;
  outputHandlers.add(make_shared<JsParakeetOutputHandler>());

  unique_ptr<OutputCallBackInterface> callback = make_unique<OutputCallBackJs>(
      env,
      args.get(0, "jsHandle"),
      args.getFunction(2, "outputCallback"),
      std::move(outputHandlers));

  auto addon = make_unique<AddonJs>(env, std::move(callback), std::move(model));
  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto [type, jsInput] = JsInterface::getInput(args);

  if (type != "audio") {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Unknown input type: " + type);
  }

  vector<float> inputSamples =
      js::TypedArray<float>(env, jsInput).as<vector<float>>(env);
  return instance.runJob(any(std::move(inputSamples)));
}
JSCATCH

// ── Streaming API ────────────────────────────────────────────────────────
// Mirrors the whisper addon surface so the SDK plugin layer can treat the
// two models interchangeably. Config is a plain JS object carrying Silero
// VAD model path + VAD tuning knobs; StreamingProcessor owns the VAD
// session + background thread that drives segmentation.

inline js_value_t*
startStreaming(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto configObj = args.getJsObject(1, "config");

  StreamingProcessor::Config config;

  auto maybeVadModelPath =
      configObj.getOptionalProperty<js::String>(env, "vadModelPath");
  if (maybeVadModelPath.has_value()) {
    config.vadModelPath = maybeVadModelPath.value().as<std::string>(env);
  }
  if (config.vadModelPath.empty()) {
    throw std::runtime_error("vadModelPath is required for streaming");
  }

  auto maybeJobId = configObj.getOptionalProperty<js::Number>(env, "jobId");
  if (!maybeJobId.has_value()) {
    throw std::runtime_error("jobId is required for streaming");
  }
  const double jobIdDouble = maybeJobId.value().as<double>(env);
  if (!(jobIdDouble >= 1.0)) {
    throw std::runtime_error("jobId must be a positive integer");
  }
  config.jobId = static_cast<decltype(config.jobId)>(jobIdDouble);

  auto maybeVadThreshold =
      configObj.getOptionalProperty<js::Number>(env, "vadThreshold");
  if (maybeVadThreshold.has_value()) {
    config.vadThreshold =
        static_cast<float>(maybeVadThreshold.value().as<double>(env));
  }

  auto maybeMinSilence =
      configObj.getOptionalProperty<js::Number>(env, "minSilenceDurationMs");
  if (maybeMinSilence.has_value()) {
    config.minSilenceDurationMs =
        static_cast<int>(maybeMinSilence.value().as<double>(env));
  }

  auto maybeMinSpeech =
      configObj.getOptionalProperty<js::Number>(env, "minSpeechDurationMs");
  if (maybeMinSpeech.has_value()) {
    config.minSpeechDurationMs =
        static_cast<int>(maybeMinSpeech.value().as<double>(env));
  }

  auto maybeMaxSpeech =
      configObj.getOptionalProperty<js::Number>(env, "maxSpeechDurationS");
  if (maybeMaxSpeech.has_value()) {
    config.maxSpeechDurationS =
        static_cast<float>(maybeMaxSpeech.value().as<double>(env));
    config.maxBufferSamples =
        static_cast<int>(config.maxSpeechDurationS) * config.sampleRate;
  }

  auto maybeSpeechPad =
      configObj.getOptionalProperty<js::Number>(env, "speechPadMs");
  if (maybeSpeechPad.has_value()) {
    config.speechPadMs =
        static_cast<int>(maybeSpeechPad.value().as<double>(env));
  }

  auto maybeSamplesOverlap =
      configObj.getOptionalProperty<js::Number>(env, "samplesOverlap");
  if (maybeSamplesOverlap.has_value()) {
    config.samplesOverlap =
        static_cast<float>(maybeSamplesOverlap.value().as<double>(env));
  }

  {
    std::lock_guard lock(g_streamingMtx);

    if (g_streamingSessions.count(&instance) != 0) {
      throw std::runtime_error(
          "Streaming session already active for this instance");
    }

    auto& parakeetModel =
        dynamic_cast<ParakeetModel&>(instance.addonCpp->model.get());
    g_streamingSessions[&instance] = std::make_unique<StreamingProcessor>(
        parakeetModel,
        instance.addonCpp->outputQueue,
        config);
  }

  return js::Boolean::create(env, true);
}
JSCATCH

inline js_value_t*
appendStreamingAudio(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto [type, jsInput] = JsInterface::getInput(args);

  if (type != "audio") {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Unknown input type: " + type);
  }

  // Parakeet's append path accepts float32 PCM directly (matching runJob);
  // there's no format auto-detect like whisper because the SDK wrapper
  // already normalises to Float32Array before forwarding.
  std::vector<float> samples =
      js::TypedArray<float>(env, jsInput).as<std::vector<float>>(env);

  if (samples.empty()) {
    return js::Boolean::create(env, false);
  }

  StreamingProcessor* processor = nullptr;
  {
    std::lock_guard lock(g_streamingMtx);
    auto it = g_streamingSessions.find(&instance);
    if (it == g_streamingSessions.end()) {
      throw std::runtime_error("No active streaming session for this instance");
    }
    processor = it->second.get();
  }

  processor->appendAudio(std::move(samples));
  return js::Boolean::create(env, true);
}
JSCATCH

inline bool
cleanupStreamingSession(
    qvac_lib_inference_addon_cpp::AddonJs& instance, bool forceful = false) {
  std::unique_ptr<StreamingProcessor> processor;
  {
    std::lock_guard lock(g_streamingMtx);
    auto it = g_streamingSessions.find(&instance);
    if (it == g_streamingSessions.end()) {
      return false;
    }
    processor = std::move(it->second);
    g_streamingSessions.erase(it);
  }
  if (forceful) {
    processor->cancel();
  } else {
    processor->end();
  }
  return true;
}

inline js_value_t*
endStreaming(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  bool cleaned = cleanupStreamingSession(instance, false);
  return js::Boolean::create(env, cleaned);
}
JSCATCH

} // namespace qvac_lib_infer_parakeet
