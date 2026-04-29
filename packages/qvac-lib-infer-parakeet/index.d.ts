/// <reference types="node" />

import type { QvacResponse } from '@qvac/infer-base';
import type { LoggerInterface } from '@qvac/logging';

/**
 * Model type options for Parakeet
 */
export type ModelType = 'tdt' | 'ctc' | 'eou' | 'sortformer';

/**
 * Parakeet-specific configuration options
 */
export interface ParakeetConfig {
  /** Model type: 'tdt' (multilingual), 'ctc' (English), 'eou' (streaming), 'sortformer' (diarization) */
  modelType?: ModelType;
  /** Maximum CPU threads for inference */
  maxThreads?: number;
  /** Enable GPU acceleration (CUDA/CoreML/DirectML) */
  useGPU?: boolean;
  /** Audio sample rate in Hz (default: 16000) */
  sampleRate?: number;
  /** Number of audio channels (default: 1, must be mono) */
  channels?: number;
  /** Enable caption/subtitle mode (default: false) */
  captionEnabled?: boolean;
  /** Include timestamps in output (default: true) */
  timestampsEnabled?: boolean;
  /** Random seed for reproducibility (-1 for random, default: -1) */
  seed?: number;
  /** Silero VAD tuning for runStreaming(). Ignored by run(). */
  vad_params?: ParakeetVadParams;
}

/**
 * Map of model file paths supplied to TranscriptionParakeet
 */
export interface TranscriptionParakeetFiles {
  /** Absolute path to TDT encoder-model.onnx */
  encoder?: string;
  /** Absolute path to TDT encoder-model.onnx.data */
  encoderData?: string;
  /** Absolute path to TDT decoder_joint-model.onnx */
  decoder?: string;
  /** Absolute path to TDT vocab.txt */
  vocab?: string;
  /** Absolute path to TDT preprocessor.onnx */
  preprocessor?: string;
  /** Absolute path to CTC model.onnx */
  model?: string;
  /** Absolute path to CTC model.onnx_data */
  modelData?: string;
  /** Absolute path to CTC/EOU tokenizer.json */
  tokenizer?: string;
  /** Absolute path to EOU encoder.onnx */
  eouEncoder?: string;
  /** Absolute path to EOU decoder_joint.onnx */
  eouDecoder?: string;
  /** Absolute path to sortformer.onnx */
  sortformer?: string;
  /** Absolute path to silero_vad.onnx — required by runStreaming() */
  vadModel?: string;
}

/**
 * Silero VAD tuning parameters for streaming mode. Field names match
 * whisper's vad_params for drop-in compatibility with SDK callers.
 */
export interface ParakeetVadParams {
  threshold?: number;
  min_silence_duration_ms?: number;
  min_speech_duration_ms?: number;
  max_speech_duration_s?: number;
  speech_pad_ms?: number;
  samples_overlap?: number;
}

/**
 * Options accepted by the TranscriptionParakeet constructor
 */
export interface TranscriptionParakeetArgs {
  /** Map of model file paths */
  files?: TranscriptionParakeetFiles;
  /** Parakeet inference configuration */
  config?: TranscriptionParakeetConfig;
  /** Optional structured logger */
  logger?: LoggerInterface;
  /** Whether to run exclusively (default: true) */
  exclusiveRun?: boolean;
  /** Additional arguments */
  [key: string]: unknown;
}

/**
 * Configuration for TranscriptionParakeet (non-path settings only)
 */
export interface TranscriptionParakeetConfig {
  /** Enable statistics collection */
  enableStats?: boolean;
  /** Parakeet-specific configuration */
  parakeetConfig?: ParakeetConfig;
  /** Additional configuration */
  [key: string]: unknown;
}

/**
 * Transcription segment returned by the model
 */
export interface TranscriptionSegment {
  /** Transcribed text */
  text: string;
  /** Start time in seconds */
  start: number;
  /** End time in seconds */
  end: number;
  /** Whether to append to previous output */
  toAppend: boolean;
  /** Segment ID */
  id?: number;
}

/**
 * Output callback events
 */
export type OutputEvent = 'JobStarted' | 'Output' | 'JobEnded' | 'Error';

/**
 * Input types accepted by the Parakeet addon
 */
export type AppendInput =
  | { type: 'audio'; data: ArrayBuffer; priority?: number }
  | { type: 'end of job' };

/**
 * Minimal interface for the native addon
 */
export interface Addon {
  activate(): Promise<void>;
  /** Returns the JS-owned job ID for the buffered or running transcription. */
  append(input: AppendInput): Promise<number>;
  /** Cancels the matching JS-owned job when one is active or buffered. */
  cancel(jobId?: number): Promise<void>;
  loadWeights(weightsData: { filename: string; chunk: Uint8Array; completed: boolean }): Promise<void>;
  status(): Promise<string>;
  pause(): Promise<void>;
  stop(): Promise<void>;
  reload(config: ParakeetConfig): Promise<void>;
  destroyInstance(): Promise<void>;
  /** Open a Silero-VAD driven streaming session. */
  startStreaming(config: {
    vadModelPath: string;
    vadThreshold?: number;
    minSilenceDurationMs?: number;
    minSpeechDurationMs?: number;
    maxSpeechDurationS?: number;
    speechPadMs?: number;
    samplesOverlap?: number;
  }): void;
  /** Push a Float32Array of PCM samples into the active streaming session. */
  appendStreamingAudio(data: { input: Float32Array }): boolean;
  /** Close the active streaming session and flush any buffered audio. */
  endStreaming(): boolean;
}

/**
 * ONNX Runtime client implementation for the Parakeet speech-to-text model.
 * Supports NVIDIA Parakeet ASR models in ONNX format.
 */
declare class TranscriptionParakeet {
  protected readonly _config: TranscriptionParakeetConfig;
  protected addon!: Addon;
  protected params: ParakeetConfig;

  /**
   * Creates an instance of TranscriptionParakeet.
   * @param opts - constructor options
   */
  constructor(opts: TranscriptionParakeetArgs);

  /**
   * Validate that required model files exist
   */
  validateModelFiles(): void;

  /**
   * Load model and activate addon.
   */
  protected _load(): Promise<void>;

  /**
   * Load model and activate addon.
   */
  load(): Promise<void>;

  /**
   * Run transcription on an audio stream.
   * When `opts.stats` was set on construction, `response.stats` matches {@link TranscriptionParakeet.RuntimeStats}.
   * @param audioStream - Stream of audio data (16kHz mono)
   * @returns A QvacResponse representing the transcription job
   */
  run(
    audioStream: AsyncIterable<Buffer>
  ): Promise<QvacResponse<TranscriptionParakeet.ParakeetRunOutput>>;

  /**
   * Run a live streaming transcription session backed by Silero VAD.
   * Requires `files.vadModel` at construction time. The caller drives the
   * PCM feed via the AsyncIterable and each VAD-detected segment is
   * transcribed by the Parakeet offline recognizer.
   */
  runStreaming(
    audioStream: AsyncIterable<Buffer>
  ): Promise<QvacResponse<TranscriptionParakeet.ParakeetRunOutput>>;

  /**
   * Reload the model with new configuration parameters.
   * @param newConfig - New configuration parameters
   */
  reload(newConfig?: {
    parakeetConfig?: Partial<ParakeetConfig>;
  }): Promise<void>;

  /**
   * Unload the model and free resources.
   */
  unload(): Promise<void>;

  /**
   * Returns the current state of the instance.
   */
  getState(): { configLoaded: boolean; weightsLoaded: boolean; destroyed: boolean };

  /**
   * Cancel the current job.
   */
  cancel(): Promise<void>;

  /**
   * Get the current status of the addon.
   */
  status(): Promise<string | undefined>;

  /**
   * Pause inference.
   */
  pause(): Promise<void>;

  /**
   * Resume inference.
   */
  unpause(): Promise<void>;

  /**
   * Destroy the instance and free all resources.
   */
  destroy(): Promise<void>;
}

declare namespace TranscriptionParakeet {
  /**
   * Keys returned by the native addon `ParakeetModel::runtimeStats()` when stats are enabled.
   * `totalTime` is wall time in seconds; `audioDurationMs` and other `*Ms` fields are milliseconds where applicable.
   */
  export interface RuntimeStats {
    totalTime: number
    realTimeFactor: number
    tokensPerSecond: number
    msPerToken: number
    audioDurationMs: number
    totalSamples: number
    totalTokens: number
    totalTranscriptions: number
    processCalls: number
    modelLoadMs: number
    melSpecMs: number
    encoderMs: number
    decoderMs: number
    totalWallMs: number
    totalMelFrames: number
    totalEncodedFrames: number
  }

  /**
   * Payload passed to `onUpdate` for transcription output (segment array or a single segment).
   */
  export type ParakeetRunOutput = TranscriptionSegment[] | TranscriptionSegment

  export {
    TranscriptionParakeet as default,
    TranscriptionParakeet,
    ModelType,
    ParakeetConfig,
    TranscriptionParakeetFiles,
    TranscriptionParakeetArgs,
    TranscriptionParakeetConfig,
    TranscriptionSegment,
    OutputEvent,
    AppendInput,
    Addon
  };
}

export = TranscriptionParakeet;

