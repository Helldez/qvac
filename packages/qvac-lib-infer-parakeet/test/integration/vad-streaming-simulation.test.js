'use strict'

/**
 * VAD Streaming Simulation Tests
 *
 * Exercises `TranscriptionParakeet.runStreaming(audioStream)` — the
 * Silero-VAD-driven simulated streaming API. The test pumps a recorded
 * 16 kHz s16le PCM clip into the model in real-time-paced chunks and
 * verifies:
 *
 *   1. The native StreamingProcessor session opens and closes cleanly.
 *   2. At least one final segment (`isPartial=false`) reaches the JS
 *      consumer when a real model and audio sample are present.
 *   3. When `partial_decode_interval_ms` is set, the consumer also
 *      observes mid-segment partial frames (`isPartial=true`).
 *   4. Cancel re-entrancy: a second `runStreaming()` immediately after
 *      a cancelled session does not throw "Streaming session already
 *      active".
 *
 * The test gracefully skips when the TDT model, the Silero VAD weights,
 * or the audio sample are missing from the local checkout, so it can
 * coexist with the existing addon-level live-stream test on machines
 * that lack the optional artifacts.
 */

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const { Readable } = require('bare-stream')

const {
  TranscriptionParakeet,
  setupJsLogger,
  binding,
  getTestPaths,
  ensureModel,
  isMobile
} = require('./helpers.js')

const { modelPath, samplesDir } = getTestPaths()

// Silero VAD weights are a small (~2 MB) optional artifact. Tests that need
// the streaming path resolve them next to the TDT model dir; CI is expected
// to drop the file there before invoking the integration suite.
const VAD_MODEL_FILENAME = 'silero_vad.onnx'

function resolveVadModelPath () {
  const candidates = [
    path.join(modelPath, VAD_MODEL_FILENAME),
    path.join(samplesDir, VAD_MODEL_FILENAME),
    path.resolve(__dirname, '../../models', VAD_MODEL_FILENAME)
  ]
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
  }
  return null
}

function loadAudioSampleAsFloat32 () {
  const samplePath = path.join(samplesDir, 'sample.raw')
  if (!fs.existsSync(samplePath)) return null
  const raw = fs.readFileSync(samplePath)
  const pcm = new Int16Array(raw.buffer, raw.byteOffset, raw.length / 2)
  const audio = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) audio[i] = pcm[i] / 32768.0
  return audio
}

// Pushable AsyncIterable that paces Float32 PCM chunks at a near-real-time
// cadence so the VAD has the chance to emit partials between writes.
function createPacedFloat32Stream (audioData, chunkDurationMs = 200, chunkDelayMs = 5) {
  const sampleRate = 16000
  const samplesPerChunk = Math.floor((chunkDurationMs / 1000) * sampleRate)
  const stream = new Readable({ read (cb) { cb(null) } })

  ;(async () => {
    for (let i = 0; i < audioData.length; i += samplesPerChunk) {
      const end = Math.min(i + samplesPerChunk, audioData.length)
      stream.push(audioData.slice(i, end))
      if (end < audioData.length && chunkDelayMs > 0) {
        await new Promise(resolve => setTimeout(resolve, chunkDelayMs))
      }
    }
    stream.push(null)
  })().catch(err => stream.destroy(err))

  return stream
}

function buildTdtFiles () {
  return {
    encoder: path.join(modelPath, 'encoder-model.onnx'),
    encoderData: path.join(modelPath, 'encoder-model.onnx.data'),
    decoder: path.join(modelPath, 'decoder_joint-model.onnx'),
    vocab: path.join(modelPath, 'vocab.txt'),
    preprocessor: path.join(modelPath, 'preprocessor.onnx')
  }
}

test('runStreaming: emits final segments and clean shutdown', { timeout: 300000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }

  const loggerBinding = setupJsLogger(binding)

  await ensureModel(modelPath)

  const vadModelPath = resolveVadModelPath()
  if (!vadModelPath) {
    loggerBinding.releaseLogger()
    t.pass(`Test skipped — ${VAD_MODEL_FILENAME} not found`)
    return
  }

  const audio = loadAudioSampleAsFloat32()
  if (!audio) {
    loggerBinding.releaseLogger()
    t.pass('Test skipped — sample.raw not found')
    return
  }

  const files = buildTdtFiles()
  files.vadModel = vadModelPath

  let model
  try {
    model = new TranscriptionParakeet({
      files,
      config: {
        parakeetConfig: {
          modelType: 'tdt',
          maxThreads: 4,
          useGPU: false
        }
      }
    })

    await model.load()

    const segments = []
    const stream = createPacedFloat32Stream(audio)
    const response = await model.runStreaming(stream)

    await response
      .onUpdate((outputArr) => {
        const items = Array.isArray(outputArr) ? outputArr : [outputArr]
        for (const seg of items) {
          if (seg && typeof seg.text === 'string') segments.push(seg)
        }
      })
      .await()

    t.ok(segments.length >= 1, 'received at least one segment from runStreaming')
    t.ok(
      segments.every(s => typeof s.text === 'string'),
      'every segment carries a text string'
    )
    t.ok(
      segments.some(s => s.isPartial !== true),
      'at least one segment is final (isPartial !== true)'
    )
  } catch (err) {
    console.error('runStreaming test failed:', err)
    t.fail(err.message)
  } finally {
    if (model) await model.destroy()
    loggerBinding.releaseLogger()
  }
})

test('runStreaming: emits mid-segment partials when interval is set', { timeout: 300000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }

  const loggerBinding = setupJsLogger(binding)

  await ensureModel(modelPath)

  const vadModelPath = resolveVadModelPath()
  if (!vadModelPath) {
    loggerBinding.releaseLogger()
    t.pass(`Test skipped — ${VAD_MODEL_FILENAME} not found`)
    return
  }

  const audio = loadAudioSampleAsFloat32()
  if (!audio) {
    loggerBinding.releaseLogger()
    t.pass('Test skipped — sample.raw not found')
    return
  }

  const files = buildTdtFiles()
  files.vadModel = vadModelPath

  let model
  try {
    model = new TranscriptionParakeet({
      files,
      config: {
        parakeetConfig: {
          modelType: 'tdt',
          maxThreads: 4,
          useGPU: false,
          // Tight cadence + low silence gate keeps the VAD segment open long
          // enough for the partial loop to fire on a short clip.
          vad_params: {
            partial_decode_interval_ms: 500,
            min_silence_duration_ms: 800
          }
        }
      }
    })

    await model.load()

    const stream = createPacedFloat32Stream(audio, 100, 10)
    const response = await model.runStreaming(stream)

    let sawPartial = false
    let sawFinal = false

    await response
      .onUpdate((outputArr) => {
        const items = Array.isArray(outputArr) ? outputArr : [outputArr]
        for (const seg of items) {
          if (!seg || typeof seg.text !== 'string') continue
          if (seg.isPartial === true) sawPartial = true
          else sawFinal = true
        }
      })
      .await()

    t.ok(sawFinal, 'received at least one final segment')
    t.ok(sawPartial, 'received at least one partial segment with partial_decode_interval_ms=500')
  } catch (err) {
    console.error('runStreaming partials test failed:', err)
    t.fail(err.message)
  } finally {
    if (model) await model.destroy()
    loggerBinding.releaseLogger()
  }
})

test('runStreaming: cancel does not block the next session', { timeout: 180000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }

  const loggerBinding = setupJsLogger(binding)

  await ensureModel(modelPath)

  const vadModelPath = resolveVadModelPath()
  if (!vadModelPath) {
    loggerBinding.releaseLogger()
    t.pass(`Test skipped — ${VAD_MODEL_FILENAME} not found`)
    return
  }

  const audio = loadAudioSampleAsFloat32()
  if (!audio) {
    loggerBinding.releaseLogger()
    t.pass('Test skipped — sample.raw not found')
    return
  }

  const files = buildTdtFiles()
  files.vadModel = vadModelPath

  let model
  try {
    model = new TranscriptionParakeet({
      files,
      config: {
        parakeetConfig: {
          modelType: 'tdt',
          maxThreads: 4,
          useGPU: false
        }
      }
    })

    await model.load()

    // First session: cancel mid-flight before the stream ends.
    const firstStream = createPacedFloat32Stream(audio, 200, 50)
    const firstResponse = await model.runStreaming(firstStream)

    // Schedule a cancel shortly after the stream starts pumping.
    setTimeout(() => { model.cancel?.().catch(() => {}) }, 200)

    try {
      await firstResponse.await()
    } catch {
      // Cancel surfaces as a job error; that's the expected path here.
    }

    // Second session: must open without "Streaming session already active".
    const segments = []
    const secondStream = createPacedFloat32Stream(audio.slice(0, audio.length / 4))
    const secondResponse = await model.runStreaming(secondStream)
    await secondResponse
      .onUpdate((outputArr) => {
        const items = Array.isArray(outputArr) ? outputArr : [outputArr]
        for (const seg of items) {
          if (seg && typeof seg.text === 'string') segments.push(seg)
        }
      })
      .await()

    t.pass('second runStreaming session opened cleanly after cancel')
    t.ok(
      segments.length === 0 || segments.every(s => typeof s.text === 'string'),
      'second session produced well-formed segments (or none, on a short clip)'
    )
  } catch (err) {
    console.error('runStreaming cancel re-entrancy test failed:', err)
    t.fail(err.message)
  } finally {
    if (model) await model.destroy()
    loggerBinding.releaseLogger()
  }
})
