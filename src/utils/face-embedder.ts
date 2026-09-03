import type * as OrtType from 'onnxruntime-web'

const EMBEDDING_INPUT_SIZE = 112
const EMBEDDING_LENGTH = 512
const DEFAULT_MODEL_URL =
  'https://static.tpsentinel.com/vendor/onnx/models/w600k_mbf.onnx'
const DEFAULT_ORT_WASM_BASE_URL =
  'https://static.tpsentinel.com/vendor/onnx/runtime-web/'

type OrtModule = typeof OrtType
type OrtSession = Pick<OrtType.InferenceSession, 'inputNames' | 'outputNames' | 'run' | 'release'>
type OrtLoader = () => Promise<OrtModule>

const defaultOrtLoader: OrtLoader = () => import('onnxruntime-web')
let activeOrtLoader: OrtLoader = defaultOrtLoader

export type FaceEmbedderOptions = {
  modelUrl?: string
  wasmBaseUrl?: string
}

export const setOrtLoader = (loader?: OrtLoader): void => {
  activeOrtLoader = loader ?? defaultOrtLoader
}

export class FaceEmbedder {
  private readonly modelUrl: string
  private readonly wasmBaseUrl: string
  private onnxRuntime: OrtModule | null = null
  private session: OrtSession | null = null
  private inputBuffer = new Float32Array(3 * EMBEDDING_INPUT_SIZE * EMBEDDING_INPUT_SIZE)

  constructor(options: FaceEmbedderOptions = {}) {
    this.modelUrl = options.modelUrl ?? DEFAULT_MODEL_URL
    this.wasmBaseUrl = ensureTrailingSlash(options.wasmBaseUrl ?? DEFAULT_ORT_WASM_BASE_URL)
  }

  async init(): Promise<void> {
    if (this.session) return

    try {
      console.info('[FaceEmbedder] Loading ONNX Runtime Web.')
      this.onnxRuntime = await activeOrtLoader()
      console.info('[FaceEmbedder] Configuring ONNX Runtime.', {
        ortVersion: this.onnxRuntime.env.versions,
        modelUrl: this.modelUrl,
        wasmBaseUrl: this.wasmBaseUrl,
      })
      this.onnxRuntime.env.wasm.wasmPaths = this.wasmBaseUrl
      this.onnxRuntime.env.wasm.numThreads = 1
      const modelBytes = await fetchModelBytes(this.modelUrl)
      console.info('[FaceEmbedder] Creating ONNX embedding session.', {
        modelUrl: this.modelUrl,
        byteLength: modelBytes.byteLength,
      })
      this.session = await this.onnxRuntime.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
      })
      console.info('[FaceEmbedder] ONNX embedding session ready.', {
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames,
      })
    } catch (error) {
      console.error('[FaceEmbedder] Failed to initialize ONNX embedding session.', {
        modelUrl: this.modelUrl,
        wasmBaseUrl: this.wasmBaseUrl,
        error,
      })
      throw error
    }
  }

  async embed(face: ImageBitmap): Promise<Float32Array> {
    if (!this.onnxRuntime || !this.session) {
      throw new Error('FaceEmbedder is not initialized. Call init() first.')
    }

    const tensor = new this.onnxRuntime.Tensor(
      'float32',
      this.createInputBuffer(face),
      [1, 3, EMBEDDING_INPUT_SIZE, EMBEDDING_INPUT_SIZE],
    )
    const outputs = await this.session.run({ [this.session.inputNames[0]!]: tensor })
    const embedding = extractEmbedding(outputs, this.session.outputNames)

    if (embedding.length !== EMBEDDING_LENGTH) {
      throw new Error(
        `Face embedding model produced ${embedding.length} values; expected ${EMBEDDING_LENGTH}.`,
      )
    }

    return normalizeL2(embedding)
  }

  async close(): Promise<void> {
    if (!this.session) return

    await this.session.release()
    this.session = null
  }

  private createInputBuffer(face: ImageBitmap): Float32Array {
    const canvas = document.createElement('canvas')
    canvas.width = EMBEDDING_INPUT_SIZE
    canvas.height = EMBEDDING_INPUT_SIZE
    const context = canvas.getContext('2d', { willReadFrequently: true })

    if (!context) {
      throw new Error('Could not create embedding canvas context.')
    }

    context.drawImage(face, 0, 0, EMBEDDING_INPUT_SIZE, EMBEDDING_INPUT_SIZE)
    const pixels = context.getImageData(0, 0, EMBEDDING_INPUT_SIZE, EMBEDDING_INPUT_SIZE).data
    const planeSize = EMBEDDING_INPUT_SIZE * EMBEDDING_INPUT_SIZE

    for (let i = 0; i < planeSize; i += 1) {
      const pixelOffset = i * 4
      this.inputBuffer[i] = (pixels[pixelOffset + 2]! - 127.5) / 127.5
      this.inputBuffer[planeSize + i] = (pixels[pixelOffset + 1]! - 127.5) / 127.5
      this.inputBuffer[planeSize * 2 + i] = (pixels[pixelOffset]! - 127.5) / 127.5
    }

    return this.inputBuffer
  }
}

export function cosineSimilarity(reference: Float32Array, probe: Float32Array): number {
  if (reference.length !== probe.length) {
    throw new Error(
      `Cannot compare embeddings with different lengths (${reference.length} and ${probe.length}).`,
    )
  }

  let similarity = 0
  for (let i = 0; i < reference.length; i += 1) {
    similarity += reference[i]! * probe[i]!
  }

  return similarity
}

function extractEmbedding(
  outputs: Awaited<ReturnType<OrtType.InferenceSession['run']>>,
  outputNames: readonly string[],
): Float32Array {
  const outputName = outputNames[0] ?? Object.keys(outputs)[0]
  if (!outputName) {
    throw new Error('Face embedding model produced no output tensors.')
  }

  const output = outputs[outputName]
  if (!output || !(output.data instanceof Float32Array)) {
    throw new Error('Face embedding output is not a float32 tensor.')
  }

  return output.data
}

async function fetchModelBytes(modelUrl: string): Promise<Uint8Array> {
  let response: Response

  try {
    response = await fetch(modelUrl)
  } catch (error) {
    console.error('[FaceEmbedder] Failed to fetch ONNX model URL.', {
      modelUrl,
      error,
    })
    throw error
  }

  const headers = {
    contentLength: response.headers.get('content-length'),
    contentType: response.headers.get('content-type'),
  }

  console.info('[FaceEmbedder] ONNX model response.', {
    modelUrl,
    ok: response.ok,
    status: response.status,
    statusText: response.statusText,
    headers,
  })

  if (!response.ok) {
    throw new Error(
      `Unable to fetch ONNX model (${response.status} ${response.statusText}) from ${modelUrl}.`,
    )
  }

  const buffer = await response.arrayBuffer()
  if (buffer.byteLength === 0) {
    throw new Error(`ONNX model response from ${modelUrl} was empty.`)
  }

  return new Uint8Array(buffer)
}

function normalizeL2(values: Float32Array): Float32Array {
  let magnitude = 0
  for (let i = 0; i < values.length; i += 1) {
    magnitude += values[i]! * values[i]!
  }

  magnitude = Math.sqrt(magnitude)
  if (!Number.isFinite(magnitude) || magnitude === 0) {
    throw new Error('Face embedding has invalid magnitude.')
  }

  const normalized = new Float32Array(values.length)
  for (let i = 0; i < values.length; i += 1) {
    normalized[i] = values[i]! / magnitude
  }

  return normalized
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith('/') ? value : `${value}/`
}
