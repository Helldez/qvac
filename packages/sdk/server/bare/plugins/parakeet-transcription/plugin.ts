import parakeetAddonLogging from "@qvac/transcription-parakeet/addonLogging";
import TranscriptionParakeet, {
  type ParakeetConfig,
  type TranscriptionParakeetFiles,
  type TranscriptionParakeetConfig,
} from "@qvac/transcription-parakeet";
import {
  definePlugin,
  defineHandler,
  defineDuplexHandler,
  transcribeRequestSchema,
  transcribeResponseSchema,
  transcribeStreamRequestSchema,
  transcribeStreamResponseSchema,
  ModelType,
  parakeetConfigSchema,
  ADDON_PARAKEET,
  type ModelSrcInput,
  type CreateModelParams,
  type PluginModelResult,
  type ResolveContext,
  type ResolveResult,
} from "@/schemas";
import { createStreamLogger, registerAddonLogger } from "@/logging";
import { parseModelPath } from "@/server/utils";
import {
  ModelLoadFailedError,
  ParakeetArtifactsRequiredError,
} from "@/utils/errors-server";
import FilesystemDL from "@qvac/dl-filesystem";
import { transcribe, transcribeStream } from "@/server/bare/ops/transcribe";
import { attachModelExecutionMs } from "@/profiling/model-execution";

type ParakeetModelConfig = {
  modelType?: string;
  maxThreads?: number;
  useGPU?: boolean;
  sampleRate?: number;
  channels?: number;
  captionEnabled?: boolean;
  timestampsEnabled?: boolean;
  // TDT
  parakeetEncoderSrc?: ModelSrcInput;
  parakeetEncoderDataSrc?: ModelSrcInput;
  parakeetDecoderSrc?: ModelSrcInput;
  parakeetVocabSrc?: ModelSrcInput;
  parakeetPreprocessorSrc?: ModelSrcInput;
  // CTC
  parakeetCtcModelSrc?: ModelSrcInput;
  parakeetCtcModelDataSrc?: ModelSrcInput;
  parakeetTokenizerSrc?: ModelSrcInput;
  // Sortformer
  parakeetSortformerSrc?: ModelSrcInput;
  // Silero VAD — streaming only
  vadModelSrc?: ModelSrcInput;
  vad_params?: {
    threshold?: number;
    min_silence_duration_ms?: number;
    min_speech_duration_ms?: number;
    max_speech_duration_s?: number;
    speech_pad_ms?: number;
    samples_overlap?: number;
  };
};


async function resolveTdtConfig(
  cfg: ParakeetModelConfig,
  ctx: ResolveContext,
): Promise<ResolveResult<ParakeetModelConfig>> {
  const {
    parakeetEncoderSrc,
    parakeetEncoderDataSrc,
    parakeetDecoderSrc,
    parakeetVocabSrc,
    parakeetPreprocessorSrc,
  } = cfg;

  if (
    !parakeetEncoderSrc ||
    !parakeetDecoderSrc ||
    !parakeetVocabSrc ||
    !parakeetPreprocessorSrc
  ) {
    throw new ParakeetArtifactsRequiredError(
      "TDT requires: parakeetEncoderSrc, parakeetDecoderSrc, parakeetVocabSrc, parakeetPreprocessorSrc",
    );
  }

  const resolve = ctx.resolveModelPath;
  const [
    encoderPath,
    encoderDataPath,
    decoderPath,
    vocabPath,
    preprocessorPath,
  ] = await Promise.all([
    resolve(parakeetEncoderSrc),
    parakeetEncoderDataSrc ? resolve(parakeetEncoderDataSrc) : undefined,
    resolve(parakeetDecoderSrc),
    resolve(parakeetVocabSrc),
    resolve(parakeetPreprocessorSrc),
  ]);

  return {
    config: cfg,
    artifacts: {
      encoder: encoderPath,
      ...(encoderDataPath !== undefined && { encoderData: encoderDataPath }),
      ...(decoderPath !== undefined && { decoder: decoderPath }),
      ...(vocabPath !== undefined && { vocab: vocabPath }),
      ...(preprocessorPath !== undefined && { preprocessor: preprocessorPath }),
    },
  };
}

async function resolveCtcConfig(
  cfg: ParakeetModelConfig,
  ctx: ResolveContext,
): Promise<ResolveResult<ParakeetModelConfig>> {
  const { parakeetCtcModelSrc, parakeetCtcModelDataSrc, parakeetTokenizerSrc } =
    cfg;

  if (!parakeetCtcModelSrc || !parakeetTokenizerSrc) {
    throw new ParakeetArtifactsRequiredError(
      "CTC requires: parakeetCtcModelSrc, parakeetTokenizerSrc",
    );
  }

  const resolve = ctx.resolveModelPath;
  const [ctcModelPath, ctcModelDataPath, tokenizerPath] = await Promise.all([
    resolve(parakeetCtcModelSrc),
    parakeetCtcModelDataSrc ? resolve(parakeetCtcModelDataSrc) : undefined,
    resolve(parakeetTokenizerSrc),
  ]);

  return {
    config: cfg,
    artifacts: {
      model: ctcModelPath,
      ...(ctcModelDataPath !== undefined && { modelData: ctcModelDataPath }),
      ...(tokenizerPath !== undefined && { tokenizer: tokenizerPath }),
    },
  };
}

async function resolveSortformerConfig(
  cfg: ParakeetModelConfig,
  ctx: ResolveContext,
): Promise<ResolveResult<ParakeetModelConfig>> {
  const { parakeetSortformerSrc } = cfg;

  if (!parakeetSortformerSrc) {
    throw new ParakeetArtifactsRequiredError(
      "Sortformer requires: parakeetSortformerSrc",
    );
  }

  const resolve = ctx.resolveModelPath;
  const sortformerPath = await resolve(parakeetSortformerSrc);

  return {
    config: cfg,
    artifacts: {
      ...(sortformerPath !== undefined && { sortformer: sortformerPath }),
    },
  };
}

function createParakeetModel(
  params: CreateModelParams,
  primaryFileKey: keyof TranscriptionParakeetFiles,
): PluginModelResult {
  const config = (params.modelConfig ?? {}) as ParakeetModelConfig;
  const artifacts = { ...(params.artifacts ?? {}) };
  const modelType = config.modelType ?? "tdt";
  const primaryPath = artifacts[primaryFileKey] ?? params.modelPath;

  if (!primaryPath) {
    throw new ModelLoadFailedError(
      `Parakeet ${modelType} requires a model source`,
    );
  }

  const { dirPath } = parseModelPath(primaryPath);
  const loader = new FilesystemDL({ dirPath });
  const logger = createStreamLogger(params.modelId, ModelType.parakeetTranscription);
  registerAddonLogger(params.modelId, ModelType.parakeetTranscription, logger);

  const files: TranscriptionParakeetFiles = {
    [primaryFileKey]: primaryPath,
    ...artifacts,
  };

  const addonConfig: TranscriptionParakeetConfig = {
    enableStats: true,
    parakeetConfig: {
      modelType,
      maxThreads: config.maxThreads,
      useGPU: config.useGPU,
      sampleRate: config.sampleRate,
      channels: config.channels,
      captionEnabled: config.captionEnabled,
      timestampsEnabled: config.timestampsEnabled,
      ...(config.vad_params && { vad_params: config.vad_params }),
    } as ParakeetConfig,
  };

  const model = new TranscriptionParakeet({
    files,
    config: addonConfig,
    logger,
  });

  return { model, loader };
}

export const parakeetPlugin = definePlugin({
  modelType: ModelType.parakeetTranscription,
  displayName: "Parakeet (NVIDIA NeMo ONNX)",
  addonPackage: ADDON_PARAKEET,
  loadConfigSchema: parakeetConfigSchema,
  skipPrimaryModelPathValidation: true,

  async resolveConfig(
    cfg: ParakeetModelConfig,
    ctx: ResolveContext,
  ): Promise<ResolveResult<ParakeetModelConfig>> {
    const modelType = cfg.modelType ?? "tdt";

    let base: ResolveResult<ParakeetModelConfig>;
    if (modelType === "ctc") base = await resolveCtcConfig(cfg, ctx);
    else if (modelType === "sortformer")
      base = await resolveSortformerConfig(cfg, ctx);
    else base = await resolveTdtConfig(cfg, ctx);

    // Silero VAD is orthogonal to the recognizer variant: resolve it once
    // and attach the path to the artifacts so createParakeetModel can pass
    // it through to the native addon.
    if (cfg.vadModelSrc) {
      const vadModelPath = await ctx.resolveModelPath(cfg.vadModelSrc);
      base = {
        ...base,
        artifacts: {
          ...(base.artifacts ?? {}),
          ...(vadModelPath !== undefined && { vadModel: vadModelPath }),
        },
      };
    }

    return base;
  },

  createModel(params: CreateModelParams): PluginModelResult {
    const modelType =
      ((params.modelConfig ?? {}) as ParakeetModelConfig).modelType ?? "tdt";

    if (modelType === "ctc") return createParakeetModel(params, "model");
    if (modelType === "sortformer")
      return createParakeetModel(params, "sortformer");
    return createParakeetModel(params, "encoder");
  },

  handlers: {
    transcribe: defineHandler({
      requestSchema: transcribeRequestSchema,
      responseSchema: transcribeResponseSchema,
      streaming: true,

      handler: async function* (request) {
        const stream = transcribe({
          modelId: request.modelId,
          audioChunk: request.audioChunk,
          prompt: request.prompt,
        });

        try {
          let result = await stream.next();
          while (!result.done) {
            yield {
              type: "transcribe" as const,
              text: result.value,
            };
            result = await stream.next();
          }

          const { modelExecutionMs, stats } = result.value;
          yield attachModelExecutionMs({
            type: "transcribe" as const,
            text: "",
            done: true,
            ...(stats && { stats }),
          }, modelExecutionMs);
        } finally {
          await stream.return?.(undefined as never);
        }
      },
    }),

    transcribeStream: defineDuplexHandler({
      requestSchema: transcribeStreamRequestSchema,
      responseSchema: transcribeStreamResponseSchema,
      streaming: true,
      duplex: true,

      handler: async function* (request, inputStream) {
        for await (const segment of transcribeStream(
          request.modelId,
          inputStream,
          request.prompt,
        )) {
          yield {
            type: "transcribeStream" as const,
            text: segment.text,
            isPartial: segment.isPartial,
          };
        }
        yield {
          type: "transcribeStream" as const,
          text: "",
          done: true,
        };
      },
    }),
  },

  logging: {
    module: parakeetAddonLogging,
    namespace: ModelType.parakeetTranscription,
  },
});
