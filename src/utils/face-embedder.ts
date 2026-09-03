import type * as OrtType from 'onnxruntime-web'

const EMBEDDING_INPUT_SIZE = 112
const DEFAULT_ARCFACE_MODEL_URL =
  'https://static.tpsentinel.com/vendor/onnx/models/w600k_mbf.onnx'
const DEFAULT_SFACE_MODEL_URL =
  'https://static.tpsentinel.com/vendor/onnx/models/face_recognition_sface_2021dec_int8.onnx'
const DEFAULT_ORT_WASM_BASE_URL =
  'https://static.tpsentinel.com/vendor/onnx/runtime-web/'

type OrtModule = typeof OrtType
type OrtSession = Pick<OrtType.InferenceSession, 'inputNames' | 'outputNames' | 'run' | 'release'>
type OrtLoader = () => Promise<OrtModule>

const defaultOrtLoader: OrtLoader = () => import('onnxruntime-web')
let activeOrtLoader: OrtLoader = defaultOrtLoader

export type FaceEmbedderOptions = {
  modelUrls?: Partial<Record<FaceEmbeddingModel, string>>
  wasmBaseUrl?: string
}
export type FaceEmbeddingModel = 'arcface-mbf' | 'sface'
export type FaceEmbeddingProfile = {
  model: FaceEmbeddingModel
  modelUrl: string
  embeddingLength: number
}

export const setOrtLoader = (loader?: OrtLoader): void => {
  activeOrtLoader = loader ?? defaultOrtLoader
}

export class FaceEmbedder {
  private readonly modelUrls: Record<FaceEmbeddingModel, string>
  private readonly wasmBaseUrl: string
  private onnxRuntime: OrtModule | null = null
  private sessions = new Map<FaceEmbeddingModel, OrtSession>()
  private inputBuffer = new Float32Array(3 * EMBEDDING_INPUT_SIZE * EMBEDDING_INPUT_SIZE)

  constructor(options: FaceEmbedderOptions = {}) {
    this.modelUrls = {
      'arcface-mbf': options.modelUrls?.['arcface-mbf'] ?? DEFAULT_ARCFACE_MODEL_URL,
      sface: options.modelUrls?.sface ?? DEFAULT_SFACE_MODEL_URL,
    }
    this.wasmBaseUrl = ensureTrailingSlash(options.wasmBaseUrl ?? DEFAULT_ORT_WASM_BASE_URL)
  }

  async init(): Promise<void> {
    if (this.onnxRuntime) return

    try {
      console.info('[FaceEmbedder] Loading ONNX Runtime Web.')
      this.onnxRuntime = await activeOrtLoader()
      console.info('[FaceEmbedder] Configuring ONNX Runtime.', {
        ortVersion: this.onnxRuntime.env.versions,
        modelUrls: this.modelUrls,
        wasmBaseUrl: this.wasmBaseUrl,
      })
      this.onnxRuntime.env.wasm.wasmPaths = this.wasmBaseUrl
      this.onnxRuntime.env.wasm.numThreads = 1
    } catch (error) {
      console.error('[FaceEmbedder] Failed to initialize ONNX embedding session.', {
        modelUrls: this.modelUrls,
        wasmBaseUrl: this.wasmBaseUrl,
        error,
      })
      throw error
    }
  }

  async embed(face: ImageBitmap, model: FaceEmbeddingModel): Promise<Float32Array> {
    if (!this.onnxRuntime) {
      throw new Error('FaceEmbedder is not initialized. Call init() first.')
    }

    const profile = getEmbeddingProfile(model, this.modelUrls[model])
    const session = await this.getSession(profile)
    const tensor = new this.onnxRuntime.Tensor(
      'float32',
      this.createInputBuffer(face, model),
      [1, 3, EMBEDDING_INPUT_SIZE, EMBEDDING_INPUT_SIZE],
    )
    const outputs = await session.run({ [session.inputNames[0]!]: tensor })
    const embedding = extractEmbedding(outputs, session.outputNames)

    if (embedding.length !== profile.embeddingLength) {
      throw new Error(
        `${model} produced ${embedding.length} values; expected ${profile.embeddingLength}.`,
      )
    }

    return model === 'sface' ? new Float32Array(embedding) : normalizeL2(embedding)
  }

  async close(): Promise<void> {
    const sessions = Array.from(this.sessions.values())
    this.sessions.clear()

    await Promise.all(sessions.map((session) => session.release()))
  }

  private async getSession(profile: FaceEmbeddingProfile): Promise<OrtSession> {
    if (!this.onnxRuntime) {
      throw new Error('FaceEmbedder is not initialized. Call init() first.')
    }

    const cachedSession = this.sessions.get(profile.model)
    if (cachedSession) return cachedSession

    const modelBytes = await fetchModelBytes(profile.modelUrl)
    console.info('[FaceEmbedder] Creating ONNX embedding session.', {
      model: profile.model,
      modelUrl: profile.modelUrl,
      byteLength: modelBytes.byteLength,
    })
    const session = await this.onnxRuntime.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
    })
    console.info('[FaceEmbedder] ONNX embedding session ready.', {
      model: profile.model,
      inputNames: session.inputNames,
      outputNames: session.outputNames,
    })
    this.sessions.set(profile.model, session)
    return session
  }

  private createInputBuffer(face: ImageBitmap, model: FaceEmbeddingModel): Float32Array {
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
      const blue = pixels[pixelOffset + 2]!
      const green = pixels[pixelOffset + 1]!
      const red = pixels[pixelOffset]!

      if (model === 'sface') {
        this.inputBuffer[i] = blue
        this.inputBuffer[planeSize + i] = green
        this.inputBuffer[planeSize * 2 + i] = red
      } else {
        this.inputBuffer[i] = (blue - 127.5) / 127.5
        this.inputBuffer[planeSize + i] = (green - 127.5) / 127.5
        this.inputBuffer[planeSize * 2 + i] = (red - 127.5) / 127.5
      }
    }

    return this.inputBuffer
  }
}

export function getEmbeddingProfile(
  model: FaceEmbeddingModel,
  modelUrl: string,
): FaceEmbeddingProfile {
  return {
    model,
    modelUrl,
    embeddingLength: model === 'sface' ? 128 : 512,
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

export function normL2Distance(reference: Float32Array, probe: Float32Array): number {
  if (reference.length !== probe.length) {
    throw new Error(
      `Cannot compare embeddings with different lengths (${reference.length} and ${probe.length}).`,
    )
  }

  let distance = 0
  for (let i = 0; i < reference.length; i += 1) {
    const delta = reference[i]! - probe[i]!
    distance += delta * delta
  }

  return Math.sqrt(distance)
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
