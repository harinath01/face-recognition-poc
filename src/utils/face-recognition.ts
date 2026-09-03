import type { FaceLandmarkerResult } from '@mediapipe/tasks-vision'
import { FaceDetector } from './face-detector'
import {
  cosineSimilarity,
  FaceEmbedder,
  normL2Distance,
  type FaceEmbeddingModel,
} from './face-embedder'

export type FaceRecognitionModel = FaceEmbeddingModel
export type FaceRecognitionDistanceMetric = 'cosine' | 'norm_l2'

export type FaceRecognitionRunMode = 'single' | 'continuous'

export type FaceRecognitionSource =
  | { type: 'image-file'; file: File }
  | { type: 'image-blob'; blob: Blob }
  | { type: 'image-bitmap'; bitmap: ImageBitmap }
  | { type: 'video'; video: HTMLVideoElement }
  | { type: 'media-stream'; stream: MediaStream }

export type FaceRecognitionDecision = 'match' | 'uncertain' | 'mismatch'

export type FaceRecognitionStage =
  | 'source-ready'
  | 'face-detecting'
  | 'face-detected'
  | 'face-rejected'
  | 'face-cropping'
  | 'face-cropped'
  | 'embedding-running'
  | 'embedding-ready'
  | 'comparing'
  | 'completed'
  | 'failed'

export type FaceDetectionSummary = {
  confidence: number
  box: {
    x: number
    y: number
    width: number
    height: number
  }
  keypoints: Array<{
    x: number
    y: number
    label?: string
  }>
}

export type FaceRecognitionPipelineUpdate = {
  runId: string
  branch: 'reference' | 'probe'
  stage: FaceRecognitionStage
  status: 'idle' | 'running' | 'success' | 'warning' | 'error'
  timestamp: number
  durationMs?: number
  message?: string
  data?: {
    sourceImage?: ImageBitmap
    detectedFaces?: FaceDetectionSummary[]
    croppedFace?: ImageBitmap
    similarity?: number
    decision?: FaceRecognitionDecision
  }
}

export type FaceRecognitionResult = {
  runId: string
  model: FaceRecognitionModel
  distanceMetric: FaceRecognitionDistanceMetric
  similarity: number
  decision: FaceRecognitionDecision
  reference: {
    face: FaceDetectionSummary
    croppedFace: ImageBitmap
  }
  probe: {
    face: FaceDetectionSummary
    croppedFace: ImageBitmap
  }
  timings: {
    detectionMs: number
    cropMs: number
    embeddingMs: number
    compareMs: number
    totalMs: number
  }
}

export type FaceRecognitionErrorCode =
  | 'model-load-failed'
  | 'source-load-failed'
  | 'no-face-detected'
  | 'multiple-faces-detected'
  | 'face-too-small'
  | 'face-near-edge'
  | 'embedding-failed'
  | 'comparison-failed'
  | 'run-aborted'

export type FaceRecognitionError = {
  code: FaceRecognitionErrorCode
  message: string
  stage?: FaceRecognitionStage
  branch?: 'reference' | 'probe'
  cause?: unknown
}

export type FaceRecognitionOptions = {
  modelBaseUrl?: string
  wasmBaseUrl?: string
  embeddingModelUrl?: string
  sfaceModelUrl?: string
  onnxWasmBaseUrl?: string
}

export type FaceRecognitionSingleCheckOptions = {
  model: FaceRecognitionModel
  reference: FaceRecognitionSource
  probe: FaceRecognitionSource
  threshold?: {
    match: number
    mismatch: number
  }
  distanceMetric?: FaceRecognitionDistanceMetric
  onUpdate?: (update: FaceRecognitionPipelineUpdate) => void
  signal?: AbortSignal
}

export type FaceRecognitionContinuousOptions = {
  model: FaceRecognitionModel
  reference: FaceRecognitionSource
  probe:
    | { type: 'video'; video: HTMLVideoElement }
    | { type: 'media-stream'; stream: MediaStream }
  intervalMs?: number
  threshold?: {
    match: number
    mismatch: number
  }
  distanceMetric?: FaceRecognitionDistanceMetric
  onUpdate?: (update: FaceRecognitionPipelineUpdate) => void
  onResult?: (result: FaceRecognitionResult) => void
  onError?: (error: FaceRecognitionError) => void
}

export type FaceRecognitionRunHandle = {
  runId: string
  pause(): void
  resume(): void
  stop(): Promise<void>
  getState(): 'running' | 'paused' | 'stopped'
}

type Branch = 'reference' | 'probe'
type SourceFrame = {
  bitmap: ImageBitmap
  ownsBitmap: boolean
}
type Point = {
  x: number
  y: number
}
type AffineTransform = {
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
}
type AlignmentPointSet = {
  source: Point[]
  target: Point[]
}
type ProcessedFace = {
  face: FaceDetectionSummary
  croppedFace: ImageBitmap
  embedding: Float32Array
  timings: {
    sourceMs: number
    detectionMs: number
    cropMs: number
    embeddingMs: number
  }
}
type FaceAlignmentLandmarks = {
  leftEye: Point
  rightEye: Point
  nose: Point
  leftMouth: Point
  rightMouth: Point
}

const DEFAULT_THRESHOLD = {
  match: 0.65,
  mismatch: 0.55,
}
const DEFAULT_SFACE_COSINE_THRESHOLD = {
  match: 0.363,
  mismatch: 0.363,
}
const DEFAULT_SFACE_NORM_L2_THRESHOLD = {
  match: 1.128,
  mismatch: 1.128,
}

export class FaceRecognition {
  private readonly detector: FaceDetector
  private readonly embedder: FaceEmbedder

  private constructor(detector: FaceDetector, embedder: FaceEmbedder) {
    this.detector = detector
    this.embedder = embedder
  }

  static async create(options: FaceRecognitionOptions = {}): Promise<FaceRecognition> {
    const detector = new FaceDetector()
    const embedder = new FaceEmbedder({
      modelUrls: {
        'arcface-mbf': options.embeddingModelUrl,
        sface: options.sfaceModelUrl,
      },
      wasmBaseUrl: options.onnxWasmBaseUrl,
    })

    try {
      await detector.init()
      await embedder.init()
    } catch (cause) {
      console.error('[FaceRecognition] Failed to initialize face recognition models.', cause)
      detector.close()
      throw createFaceRecognitionError({
        code: 'model-load-failed',
        message: 'Unable to initialize face recognition models.',
        stage: 'failed',
        cause,
      })
    }

    return new FaceRecognition(detector, embedder)
  }

  async checkOnce(options: FaceRecognitionSingleCheckOptions): Promise<FaceRecognitionResult> {
    const runId = createRunId()
    const distanceMetric = getDistanceMetric(options.model, options.distanceMetric)
    const threshold = getThreshold(options.model, distanceMetric, options.threshold)
    const totalStartedAt = performance.now()

    try {
      this.assertNotAborted(options.signal)
      const reference = await this.processBranch({
        branch: 'reference',
        runId,
        source: options.reference,
        model: options.model,
        signal: options.signal,
        onUpdate: options.onUpdate,
      })

      this.assertNotAborted(options.signal)
      const probe = await this.processBranch({
        branch: 'probe',
        runId,
        source: options.probe,
        model: options.model,
        signal: options.signal,
        onUpdate: options.onUpdate,
      })

      const compareStartedAt = performance.now()
      options.onUpdate?.(
        createUpdate({
          runId,
          branch: 'probe',
          stage: 'comparing',
          status: 'running',
          message: 'Comparing reference and probe embeddings.',
        }),
      )
      const similarity = compareEmbeddings(reference.embedding, probe.embedding, distanceMetric)
      const compareMs = elapsed(compareStartedAt)
      const decision = decide(similarity, threshold, distanceMetric)

      options.onUpdate?.(
        createUpdate({
          runId,
          branch: 'probe',
          stage: 'completed',
          status: 'success',
          durationMs: compareMs,
          message: `Comparison complete with ${decision} decision.`,
          data: {
            similarity,
            decision,
          },
        }),
      )

      return {
        runId,
        model: options.model,
        distanceMetric,
        similarity,
        decision,
        reference: {
          face: reference.face,
          croppedFace: reference.croppedFace,
        },
        probe: {
          face: probe.face,
          croppedFace: probe.croppedFace,
        },
        timings: {
          detectionMs: reference.timings.detectionMs + probe.timings.detectionMs,
          cropMs: reference.timings.cropMs + probe.timings.cropMs,
          embeddingMs: reference.timings.embeddingMs + probe.timings.embeddingMs,
          compareMs,
          totalMs: elapsed(totalStartedAt),
        },
      }
    } catch (error) {
      const recognitionError = normalizeFaceRecognitionError(error)
      options.onUpdate?.(
        createUpdate({
          runId,
          branch: recognitionError.branch ?? 'probe',
          stage: recognitionError.stage ?? 'failed',
          status: 'error',
          message: recognitionError.message,
        }),
      )
      throw recognitionError
    }
  }

  async startContinuous(
    options: FaceRecognitionContinuousOptions,
  ): Promise<FaceRecognitionRunHandle> {
    const runId = createRunId()
    const intervalMs = options.intervalMs ?? 1000
    const distanceMetric = getDistanceMetric(options.model, options.distanceMetric)
    const threshold = getThreshold(options.model, distanceMetric, options.threshold)
    const controller = new AbortController()
    let state: 'running' | 'paused' | 'stopped' = 'running'
    let timer: number | undefined

    let reference: ProcessedFace

    try {
      reference = await this.processBranch({
        branch: 'reference',
        runId,
        source: options.reference,
        model: options.model,
        signal: controller.signal,
        onUpdate: options.onUpdate,
      })
    } catch (error) {
      const recognitionError = normalizeFaceRecognitionError(error)
      options.onError?.(recognitionError)
      throw recognitionError
    }

    const runProbeCheck = async () => {
      if (state !== 'running') return

      const probeRunId = createRunId()
      const totalStartedAt = performance.now()

      try {
        const probe = await this.processBranch({
          branch: 'probe',
          runId: probeRunId,
          source:
            options.probe.type === 'video'
              ? { type: 'video', video: options.probe.video }
              : { type: 'media-stream', stream: options.probe.stream },
          model: options.model,
          signal: controller.signal,
          onUpdate: options.onUpdate,
        })

        const compareStartedAt = performance.now()
        options.onUpdate?.(
          createUpdate({
            runId: probeRunId,
            branch: 'probe',
            stage: 'comparing',
            status: 'running',
            message: 'Comparing enrolled reference and live probe embeddings.',
          }),
        )
        const similarity = compareEmbeddings(reference.embedding, probe.embedding, distanceMetric)
        const compareMs = elapsed(compareStartedAt)
        const decision = decide(similarity, threshold, distanceMetric)

        options.onUpdate?.(
          createUpdate({
            runId: probeRunId,
            branch: 'probe',
            stage: 'completed',
            status: 'success',
            durationMs: compareMs,
            message: `Comparison complete with ${decision} decision.`,
            data: {
              similarity,
              decision,
            },
          }),
        )

        const result: FaceRecognitionResult = {
          runId: probeRunId,
          model: options.model,
          distanceMetric,
          similarity,
          decision,
          reference: {
            face: reference.face,
            croppedFace: reference.croppedFace,
          },
          probe: {
            face: probe.face,
            croppedFace: probe.croppedFace,
          },
          timings: {
            detectionMs: probe.timings.detectionMs,
            cropMs: probe.timings.cropMs,
            embeddingMs: probe.timings.embeddingMs,
            compareMs,
            totalMs: elapsed(totalStartedAt),
          },
        }
        options.onResult?.(result)
      } catch (error) {
        const recognitionError = normalizeFaceRecognitionError(error)
        if (recognitionError.code !== 'run-aborted') {
          options.onError?.(recognitionError)
        }
      } finally {
        if (state === 'running') {
          timer = window.setTimeout(runProbeCheck, intervalMs)
        }
      }
    }

    timer = window.setTimeout(runProbeCheck, 0)

    return {
      runId,
      pause() {
        if (state === 'running') {
          state = 'paused'
          if (timer) window.clearTimeout(timer)
        }
      },
      resume() {
        if (state === 'paused') {
          state = 'running'
          timer = window.setTimeout(runProbeCheck, 0)
        }
      },
      async stop() {
        state = 'stopped'
        controller.abort()
        if (timer) window.clearTimeout(timer)
      },
      getState() {
        return state
      },
    }
  }

  close(): void {
    this.detector.close()
    void this.embedder.close()
  }

  private async processBranch({
    branch,
    runId,
    source,
    model,
    signal,
    onUpdate,
  }: {
    branch: Branch
    runId: string
    source: FaceRecognitionSource
    model: FaceRecognitionModel
    signal?: AbortSignal
    onUpdate?: (update: FaceRecognitionPipelineUpdate) => void
  }): Promise<ProcessedFace> {
    const sourceStartedAt = performance.now()
    const frame = await sourceToFrame(source)
    const sourceMs = elapsed(sourceStartedAt)

    this.assertNotAborted(signal)
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'source-ready',
        status: 'success',
        durationMs: sourceMs,
        message: `${branchLabel(branch)} source is ready.`,
        data: { sourceImage: frame.bitmap },
      }),
    )

    const detectionStartedAt = performance.now()
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'face-detecting',
        status: 'running',
        message: 'Running MediaPipe face landmarker.',
      }),
    )

    this.assertNotAborted(signal)
    const landmarkResult = this.detector.detectLandmarks(frame.bitmap)
    const detectionMs = elapsed(detectionStartedAt)
    const faces = mapLandmarkerResult(frame.bitmap, landmarkResult)

    if (faces.length !== 1) {
      const code = faces.length === 0 ? 'no-face-detected' : 'multiple-faces-detected'
      const message =
        faces.length === 0
          ? `${branchLabel(branch)} face was not detected.`
          : `${branchLabel(branch)} has multiple detected faces.`

      onUpdate?.(
        createUpdate({
          runId,
          branch,
          stage: 'face-rejected',
          status: 'error',
          durationMs: detectionMs,
          message,
          data: {
            sourceImage: frame.bitmap,
            detectedFaces: faces,
          },
        }),
      )

      throw createFaceRecognitionError({
        code,
        message,
        stage: 'face-rejected',
        branch,
      })
    }

    const face = faces[0]
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'face-detected',
        status: 'success',
        durationMs: detectionMs,
        message: '1 face detected.',
        data: {
          sourceImage: frame.bitmap,
          detectedFaces: faces,
        },
      }),
    )

    const cropStartedAt = performance.now()
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'face-cropping',
        status: 'running',
        message: 'Creating face crop.',
      }),
    )

    this.assertNotAborted(signal)
    const croppedFace = await cropFace(
      frame.bitmap,
      face,
      getPrimaryFaceAlignmentLandmarks(frame.bitmap, landmarkResult),
    )
    const cropMs = elapsed(cropStartedAt)
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'face-cropped',
        status: 'success',
        durationMs: cropMs,
        message: 'Face crop ready.',
        data: {
          sourceImage: frame.bitmap,
          detectedFaces: faces,
          croppedFace,
        },
      }),
    )

    const embeddingStartedAt = performance.now()
    onUpdate?.(
      createUpdate({
        runId,
        branch,
        stage: 'embedding-running',
        status: 'running',
        message: 'Running 512-d face embedding model.',
      }),
    )
    let embedding: Float32Array
    let embeddingMs = 0
    try {
      this.assertNotAborted(signal)
      embedding = await this.embedder.embed(croppedFace, model)
      embeddingMs = elapsed(embeddingStartedAt)
      onUpdate?.(
        createUpdate({
          runId,
          branch,
          stage: 'embedding-ready',
          status: 'success',
          durationMs: embeddingMs,
          message: '512-d face vector ready.',
        }),
      )
    } catch (cause) {
      embeddingMs = elapsed(embeddingStartedAt)
      onUpdate?.(
        createUpdate({
          runId,
          branch,
          stage: 'embedding-ready',
          status: 'error',
          durationMs: embeddingMs,
          message: `${branchLabel(branch)} embedding failed.`,
        }),
      )
      throw createFaceRecognitionError({
        code: 'embedding-failed',
        message: `${branchLabel(branch)} embedding failed.`,
        stage: 'embedding-ready',
        branch,
        cause,
      })
    }

    if (frame.ownsBitmap) {
      frame.bitmap.close()
    }

    return {
      face,
      croppedFace,
      embedding,
      timings: {
        sourceMs,
        detectionMs,
        cropMs,
        embeddingMs,
      },
    }
  }

  private assertNotAborted(signal?: AbortSignal) {
    if (signal?.aborted) {
      throw createFaceRecognitionError({
        code: 'run-aborted',
        message: 'Face recognition run was aborted.',
        stage: 'failed',
      })
    }
  }
}

async function sourceToFrame(source: FaceRecognitionSource): Promise<SourceFrame> {
  try {
    if (source.type === 'image-file') {
      return {
        bitmap: await createImageBitmap(source.file),
        ownsBitmap: true,
      }
    }

    if (source.type === 'image-blob') {
      return {
        bitmap: await createImageBitmap(source.blob),
        ownsBitmap: true,
      }
    }

    if (source.type === 'image-bitmap') {
      return {
        bitmap: source.bitmap,
        ownsBitmap: false,
      }
    }

    if (source.type === 'video') {
      return {
        bitmap: await createBitmapFromVideo(source.video),
        ownsBitmap: true,
      }
    }

    return {
      bitmap: await createBitmapFromMediaStream(source.stream),
      ownsBitmap: true,
    }
  } catch (cause) {
    throw createFaceRecognitionError({
      code: 'source-load-failed',
      message: 'Unable to load face recognition source.',
      stage: 'source-ready',
      cause,
    })
  }
}

async function createBitmapFromMediaStream(stream: MediaStream): Promise<ImageBitmap> {
  const video = document.createElement('video')
  video.muted = true
  video.playsInline = true
  video.srcObject = stream
  await video.play()
  await waitForVideoFrame(video)
  const bitmap = await createBitmapFromVideo(video)
  video.pause()
  video.srcObject = null
  return bitmap
}

async function createBitmapFromVideo(video: HTMLVideoElement): Promise<ImageBitmap> {
  if (!video.videoWidth || !video.videoHeight) {
    await waitForVideoFrame(video)
  }

  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Could not create canvas context.')
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return createImageBitmap(canvas)
}

function waitForVideoFrame(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return Promise.resolve()
  }

  return new Promise((resolve, reject) => {
    const handleLoadedData = () => {
      cleanup()
      resolve()
    }
    const handleError = () => {
      cleanup()
      reject(new Error('Video frame is not available.'))
    }
    const cleanup = () => {
      video.removeEventListener('loadeddata', handleLoadedData)
      video.removeEventListener('error', handleError)
    }

    video.addEventListener('loadeddata', handleLoadedData, { once: true })
    video.addEventListener('error', handleError, { once: true })
  })
}

function mapLandmarkerResult(
  image: ImageBitmap,
  result: FaceLandmarkerResult,
): FaceDetectionSummary[] {
  return result.faceLandmarks.map((landmarks) => {
    const bounds = landmarks.reduce(
      (box, landmark) => ({
        minX: Math.min(box.minX, landmark.x * image.width),
        minY: Math.min(box.minY, landmark.y * image.height),
        maxX: Math.max(box.maxX, landmark.x * image.width),
        maxY: Math.max(box.maxY, landmark.y * image.height),
      }),
      {
        minX: Number.POSITIVE_INFINITY,
        minY: Number.POSITIVE_INFINITY,
        maxX: Number.NEGATIVE_INFINITY,
        maxY: Number.NEGATIVE_INFINITY,
      },
    )
    const alignment = getFaceAlignmentLandmarksFromList(image, landmarks)

    return {
      confidence: 1,
      box: {
        x: clamp(bounds.minX, 0, image.width),
        y: clamp(bounds.minY, 0, image.height),
        width: clamp(bounds.maxX - bounds.minX, 1, image.width - bounds.minX),
        height: clamp(bounds.maxY - bounds.minY, 1, image.height - bounds.minY),
      },
      keypoints: alignment
        ? [
            normalizedKeypoint(image, alignment.leftEye, 'left eye'),
            normalizedKeypoint(image, alignment.rightEye, 'right eye'),
            normalizedKeypoint(image, alignment.nose, 'nose'),
            normalizedKeypoint(image, alignment.leftMouth, 'left mouth'),
            normalizedKeypoint(image, alignment.rightMouth, 'right mouth'),
          ]
        : [],
    }
  })
}

async function cropFace(
  image: ImageBitmap,
  face: FaceDetectionSummary,
  landmarks: FaceAlignmentLandmarks | null,
): Promise<ImageBitmap> {
  const canvas = document.createElement('canvas')
  canvas.width = 112
  canvas.height = 112
  const context = canvas.getContext('2d')

  if (!context) {
    throw createFaceRecognitionError({
      code: 'embedding-failed',
      message: 'Could not create crop canvas context.',
      stage: 'face-cropping',
    })
  }

  context.fillStyle = 'rgb(127, 127, 127)'
  context.fillRect(0, 0, canvas.width, canvas.height)

  const alignedTransform = createFaceAlignmentTransform(image, face, landmarks)

  if (alignedTransform) {
    context.setTransform(
      alignedTransform.a,
      alignedTransform.b,
      alignedTransform.c,
      alignedTransform.d,
      alignedTransform.e,
      alignedTransform.f,
    )
    context.drawImage(image, 0, 0)
    context.resetTransform()
  } else {
    const padding = Math.round(Math.max(face.box.width, face.box.height) * 0.18)
    const sx = clamp(Math.floor(face.box.x - padding), 0, image.width)
    const sy = clamp(Math.floor(face.box.y - padding), 0, image.height)
    const sw = clamp(Math.ceil(face.box.width + padding * 2), 1, image.width - sx)
    const sh = clamp(Math.ceil(face.box.height + padding * 2), 1, image.height - sy)

    context.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height)
  }

  return createImageBitmap(canvas)
}

function createFaceAlignmentTransform(
  image: ImageBitmap,
  face: FaceDetectionSummary,
  landmarks: FaceAlignmentLandmarks | null,
): AffineTransform | null {
  const points = getAlignmentPoints(image, face, landmarks)
  if (!points) return null

  return estimateSimilarityTransform(points.source, points.target)
}

function getAlignmentPoints(
  image: ImageBitmap,
  face: FaceDetectionSummary,
  landmarks: FaceAlignmentLandmarks | null,
): AlignmentPointSet | null {
  if (landmarks) {
    return {
      source: [
        landmarks.leftEye,
        landmarks.rightEye,
        landmarks.nose,
        landmarks.leftMouth,
        landmarks.rightMouth,
      ],
      target: [
        { x: 38.2946, y: 51.6963 },
        { x: 73.5318, y: 51.5014 },
        { x: 56.0252, y: 71.7366 },
        { x: 41.5493, y: 92.3655 },
        { x: 70.7299, y: 92.2041 },
      ],
    }
  }

  const absoluteKeypoints = face.keypoints.map((keypoint) => ({
    x: keypoint.x * image.width,
    y: keypoint.y * image.height,
    label: keypoint.label?.toLowerCase() ?? '',
  }))

  if (absoluteKeypoints.length >= 4) {
    const [firstEye, secondEye] = absoluteKeypoints.slice(0, 2).sort((a, b) => a.x - b.x)
    const nose = absoluteKeypoints[2]!
    const mouthCenter = absoluteKeypoints[3]!

    return {
      source: [firstEye!, secondEye!, nose, mouthCenter],
      target: [
        { x: 38.2946, y: 51.6963 },
        { x: 73.5318, y: 51.5014 },
        { x: 56.0252, y: 71.7366 },
        { x: 56.1458, y: 92.2848 },
      ],
    }
  }

  const labelledEyes = absoluteKeypoints
    .filter((point) => point.label.includes('eye'))
    .sort((a, b) => a.x - b.x)
  const labelledNose = absoluteKeypoints.find((point) => point.label.includes('nose'))

  if (labelledEyes.length >= 2 && labelledNose) {
    return {
      source: [labelledEyes[0]!, labelledEyes[1]!, labelledNose],
      target: [
        { x: 38.2946, y: 51.6963 },
        { x: 73.5318, y: 51.5014 },
        { x: 56.0252, y: 71.7366 },
      ],
    }
  }

  if (absoluteKeypoints.length < 3) return null

  const fallbackEyes = absoluteKeypoints.slice(0, 2).sort((a, b) => a.x - b.x)
  return {
    source: [fallbackEyes[0]!, fallbackEyes[1]!, absoluteKeypoints[2]!],
    target: [
      { x: 38.2946, y: 51.6963 },
      { x: 73.5318, y: 51.5014 },
      { x: 56.0252, y: 71.7366 },
    ],
  }
}

function getPrimaryFaceAlignmentLandmarks(
  image: ImageBitmap,
  result: FaceLandmarkerResult,
): FaceAlignmentLandmarks | null {
  const landmarks = result.faceLandmarks[0]
  if (!landmarks) return null

  return getFaceAlignmentLandmarksFromList(image, landmarks)
}

function getFaceAlignmentLandmarksFromList(
  image: ImageBitmap,
  landmarks: FaceLandmarkerResult['faceLandmarks'][number],
): FaceAlignmentLandmarks | null {
  const firstEye = averageLandmarks(image, landmarks, [33, 133])
  const secondEye = averageLandmarks(image, landmarks, [362, 263])
  const nose = landmarkToPoint(image, landmarks[1])
  const firstMouth = landmarkToPoint(image, landmarks[61])
  const secondMouth = landmarkToPoint(image, landmarks[291])

  if (!firstEye || !secondEye || !nose || !firstMouth || !secondMouth) return null

  const [leftEye, rightEye] =
    firstEye.x < secondEye.x ? [firstEye, secondEye] : [secondEye, firstEye]
  const [leftMouth, rightMouth] =
    firstMouth.x < secondMouth.x ? [firstMouth, secondMouth] : [secondMouth, firstMouth]

  return {
    leftEye,
    rightEye,
    nose,
    leftMouth,
    rightMouth,
  }
}

function normalizedKeypoint(
  image: ImageBitmap,
  point: Point,
  label: string,
): FaceDetectionSummary['keypoints'][number] {
  return {
    x: point.x / image.width,
    y: point.y / image.height,
    label,
  }
}

function averageLandmarks(
  image: ImageBitmap,
  landmarks: FaceLandmarkerResult['faceLandmarks'][number],
  indices: readonly number[],
): Point | null {
  const points = indices.map((index) => landmarkToPoint(image, landmarks[index]))
  if (points.some((point) => !point)) return null

  return meanPoint(points as Point[])
}

function landmarkToPoint(
  image: ImageBitmap,
  landmark: FaceLandmarkerResult['faceLandmarks'][number][number] | undefined,
): Point | null {
  if (!landmark) return null

  return {
    x: landmark.x * image.width,
    y: landmark.y * image.height,
  }
}

function estimateSimilarityTransform(
  source: readonly Point[],
  target: readonly Point[],
): AffineTransform | null {
  if (source.length !== target.length || source.length < 2) return null

  const sourceMean = meanPoint(source)
  const targetMean = meanPoint(target)
  let denominator = 0
  let alphaNumerator = 0
  let betaNumerator = 0

  for (let i = 0; i < source.length; i += 1) {
    const sx = source[i]!.x - sourceMean.x
    const sy = source[i]!.y - sourceMean.y
    const tx = target[i]!.x - targetMean.x
    const ty = target[i]!.y - targetMean.y

    denominator += sx * sx + sy * sy
    alphaNumerator += sx * tx + sy * ty
    betaNumerator += sx * ty - sy * tx
  }

  if (denominator === 0) return null

  const alpha = alphaNumerator / denominator
  const beta = betaNumerator / denominator

  return {
    a: alpha,
    b: beta,
    c: -beta,
    d: alpha,
    e: targetMean.x - alpha * sourceMean.x + beta * sourceMean.y,
    f: targetMean.y - beta * sourceMean.x - alpha * sourceMean.y,
  }
}

function meanPoint(points: readonly Point[]): Point {
  const sum = points.reduce(
    (total, point) => ({
      x: total.x + point.x,
      y: total.y + point.y,
    }),
    { x: 0, y: 0 },
  )

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
  }
}

function compareEmbeddings(
  reference: Float32Array,
  probe: Float32Array,
  distanceMetric: FaceRecognitionDistanceMetric,
): number {
  if (distanceMetric === 'norm_l2') {
    return normL2Distance(normalizeEmbedding(reference), normalizeEmbedding(probe))
  }

  return cosineSimilarity(normalizeEmbedding(reference), normalizeEmbedding(probe))
}

function getDistanceMetric(
  model: FaceRecognitionModel,
  distanceMetric?: FaceRecognitionDistanceMetric,
): FaceRecognitionDistanceMetric {
  if (model === 'sface') return distanceMetric ?? 'cosine'
  return 'cosine'
}

function getThreshold(
  model: FaceRecognitionModel,
  distanceMetric: FaceRecognitionDistanceMetric,
  threshold?: { match: number; mismatch: number },
): { match: number; mismatch: number } {
  if (threshold) return threshold
  if (model === 'sface' && distanceMetric === 'norm_l2') return DEFAULT_SFACE_NORM_L2_THRESHOLD
  if (model === 'sface') return DEFAULT_SFACE_COSINE_THRESHOLD
  return DEFAULT_THRESHOLD
}

function normalizeEmbedding(values: Float32Array): Float32Array {
  let magnitude = 0
  for (let i = 0; i < values.length; i += 1) {
    magnitude += values[i]! * values[i]!
  }

  magnitude = Math.sqrt(magnitude)
  if (!Number.isFinite(magnitude) || magnitude === 0) {
    throw createFaceRecognitionError({
      code: 'comparison-failed',
      message: 'Embedding has invalid magnitude.',
      stage: 'comparing',
    })
  }

  const normalized = new Float32Array(values.length)
  for (let i = 0; i < values.length; i += 1) {
    normalized[i] = values[i]! / magnitude
  }

  return normalized
}

function decide(
  similarity: number,
  threshold: { match: number; mismatch: number },
  distanceMetric: FaceRecognitionDistanceMetric,
): FaceRecognitionDecision {
  if (distanceMetric === 'norm_l2') {
    if (similarity <= threshold.match) return 'match'
    if (similarity > threshold.mismatch) return 'mismatch'
    return 'uncertain'
  }

  if (similarity >= threshold.match) return 'match'
  if (similarity < threshold.mismatch) return 'mismatch'
  return 'uncertain'
}

function createUpdate(
  update: Omit<FaceRecognitionPipelineUpdate, 'timestamp'>,
): FaceRecognitionPipelineUpdate {
  return {
    ...update,
    timestamp: performance.now(),
  }
}

function createRunId(): string {
  return `face-run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function createFaceRecognitionError(error: FaceRecognitionError): FaceRecognitionError {
  return error
}

function normalizeFaceRecognitionError(error: unknown): FaceRecognitionError {
  if (isFaceRecognitionError(error)) {
    return error
  }

  return {
    code: 'comparison-failed',
    message: error instanceof Error ? error.message : 'Face recognition failed.',
    stage: 'failed',
    cause: error,
  }
}

function isFaceRecognitionError(error: unknown): error is FaceRecognitionError {
  return (
    error !== null &&
    typeof error === 'object' &&
    'code' in error &&
    'message' in error
  )
}

function branchLabel(branch: Branch): string {
  return branch === 'reference' ? 'Reference' : 'Probe'
}

function elapsed(startedAt: number): number {
  return Math.round((performance.now() - startedAt) * 10) / 10
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
