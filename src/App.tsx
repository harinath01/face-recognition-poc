import { For, Show, createEffect, createMemo, createSignal, onCleanup, type JSX } from 'solid-js'
import {
  FaceRecognition,
  type FaceDetectionSummary,
  type FaceRecognitionDistanceMetric,
  type FaceRecognitionModel,
  type FaceRecognitionPipelineUpdate,
  type FaceRecognitionResult,
  type FaceRecognitionRunHandle,
  type FaceRecognitionStage,
} from './utils/face-recognition'

type ProbeMode = 'static' | 'live'
type ReferenceMode = 'static' | 'live'
type SourceDimensions = { width: number; height: number }
type PointLike = { x: number; y: number }
type PipelineRow = FaceRecognitionPipelineUpdate
type CropPreview = {
  title: string
  detail: string
  url: string
}

const STAGE_LABELS: Record<Exclude<FaceRecognitionStage, 'embedding-running' | 'embedding-ready'>, string> = {
  'source-ready': 'Image Ingest',
  'face-detecting': 'Detection',
  'face-detected': 'Detection',
  'face-rejected': 'Detection',
  'face-cropping': 'Align & Crop',
  'face-cropped': 'Align & Crop',
  comparing: 'Cosine Compare',
  completed: 'Cosine Compare',
  failed: 'Failed',
}

const MODEL_LABELS: Record<FaceRecognitionModel, string> = {
  'arcface-mbf': 'ArcFace MBF',
  sface: 'SFace',
}

export function App() {
  let referenceInputRef: HTMLInputElement | undefined
  let probeInputRef: HTMLInputElement | undefined
  let referenceVideoRef: HTMLVideoElement | undefined
  let videoRef: HTMLVideoElement | undefined
  let recognizerPromise: Promise<FaceRecognition> | undefined
  let referenceObjectUrl: string | undefined
  let probeObjectUrl: string | undefined
  let referenceCropObjectUrl: string | undefined
  let probeCropObjectUrl: string | undefined
  let activeRunId: string | undefined
  let continuousRun: FaceRecognitionRunHandle | undefined

  const [referenceFile, setReferenceFile] = createSignal<File>()
  const [probeFile, setProbeFile] = createSignal<File>()
  const [referencePreview, setReferencePreview] = createSignal<string>()
  const [probePreview, setProbePreview] = createSignal<string>()
  const [referenceCropPreview, setReferenceCropPreview] = createSignal<string>()
  const [probeCropPreview, setProbeCropPreview] = createSignal<string>()
  const [referenceFace, setReferenceFace] = createSignal<FaceDetectionSummary>()
  const [probeFace, setProbeFace] = createSignal<FaceDetectionSummary>()
  const [referenceDimensions, setReferenceDimensions] = createSignal<SourceDimensions>()
  const [probeDimensions, setProbeDimensions] = createSignal<SourceDimensions>()
  const [referenceMode, setReferenceMode] = createSignal<ReferenceMode>('static')
  const [referenceSourceDetail, setReferenceSourceDetail] = createSignal('Reference enrollment source')
  const [probeMode, setProbeMode] = createSignal<ProbeMode>('static')
  const [recognitionModel, setRecognitionModel] = createSignal<FaceRecognitionModel>('arcface-mbf')
  const [distanceMetric, setDistanceMetric] = createSignal<FaceRecognitionDistanceMetric>('cosine')
  const [cameraStream, setCameraStream] = createSignal<MediaStream>()
  const [cameraError, setCameraError] = createSignal<string>()
  const [isStartingCamera, setIsStartingCamera] = createSignal(false)
  const [isVerifying, setIsVerifying] = createSignal(false)
  const [isContinuousRunning, setIsContinuousRunning] = createSignal(false)
  const [result, setResult] = createSignal<FaceRecognitionResult>()
  const [pipelineRows, setPipelineRows] = createSignal<PipelineRow[]>([])
  const [cropPreview, setCropPreview] = createSignal<CropPreview>()

  const hasProbe = createMemo(() =>
    probeMode() === 'live' ? Boolean(cameraStream()) : Boolean(probeFile()),
  )
  const isReady = createMemo(() => Boolean(referenceFile()) && hasProbe() && !isVerifying())
  const canRunContinuous = createMemo(
    () => probeMode() === 'live' && Boolean(referenceFile()) && Boolean(cameraStream()) && !isVerifying(),
  )
  const referenceRows = createMemo(() => pipelineRows().filter((row) => row.branch === 'reference'))
  const probeRows = createMemo(() => pipelineRows().filter((row) => row.branch === 'probe'))
  const referenceTotalMs = createMemo(() => totalDuration(referenceRows()))
  const probeTotalMs = createMemo(() => totalDuration(probeRows()))
  const score = createMemo(() => result()?.similarity.toFixed(3) ?? '0.000')
  const scorePercent = createMemo(() => {
    if (result()?.distanceMetric === 'norm_l2') {
      return `${Math.max(0, Math.min(100, 100 - (Number(score()) / 1.5) * 100))}%`
    }

    return `${Math.round(Number(score()) * 1000) / 10}%`
  })
  const metricLabel = createMemo(() => {
    if (result()?.distanceMetric === 'norm_l2') return 'L2 Distance'
    return 'Cosine Score'
  })
  const vectorLabel = createMemo(() =>
    recognitionModel() === 'sface' ? '128-d Vector' : '512-d Vector',
  )
  const compareLabel = createMemo(() =>
    distanceMetric() === 'norm_l2' ? 'L2 Compare' : 'Cosine Compare',
  )
  const thresholdLabel = createMemo(() => {
    if (recognitionModel() === 'sface' && distanceMetric() === 'norm_l2') return '1.128'
    if (recognitionModel() === 'sface') return '0.363'
    return '0.650'
  })

  const resultTitle = createMemo(() => {
    if (cameraError()) return 'CAMERA UNAVAILABLE'
    if (isContinuousRunning()) return 'CONTINUOUS PIPELINE'
    if (isVerifying()) return 'PIPELINE RUNNING'
    if (result()) return result()?.decision === 'match' ? 'VERIFIED MATCH' : 'REVIEW REQUIRED'
    if (isReady()) return 'READY TO VERIFY'
    if (!referenceFile() && !hasProbe()) return 'AWAITING INPUTS'
    if (!referenceFile()) return 'REFERENCE REQUIRED'
    return 'PROBE REQUIRED'
  })

  const resultDetail = createMemo(() => {
    if (cameraError()) return cameraError()
    if (isContinuousRunning()) return 'Live probe frames are being checked continuously.'
    if (isVerifying()) return 'MediaPipe is detecting faces and preparing crops.'
    if (result()) {
      return `${metricLabel()} ${score()}. Recognition used ${MODEL_LABELS[result()!.model]}.`
    }
    if (isReady()) {
      return probeMode() === 'live'
        ? 'Live webcam frames are available for probe evaluation.'
        : 'The uploaded probe image is available for static evaluation.'
    }
    if (!referenceFile() && !hasProbe()) {
      return 'Upload a reference image, then provide a probe candidate.'
    }
    if (!referenceFile()) return 'Choose a reference image before verification.'
    return probeMode() === 'live'
      ? 'Allow webcam access to start live probe evaluation.'
      : 'Choose a static probe image or switch to webcam stream.'
  })

  createEffect(() => {
    const stream = cameraStream()

    if (videoRef) {
      videoRef.srcObject = stream ?? null
    }

    if (referenceVideoRef) {
      referenceVideoRef.srcObject = stream ?? null
    }
  })

  function getRecognizer() {
    recognizerPromise ??= FaceRecognition.create({
      modelBaseUrl: `${import.meta.env.BASE_URL}models`,
      wasmBaseUrl: `${import.meta.env.BASE_URL}wasm`,
      embeddingModelUrl: 'https://static.tpsentinel.com/vendor/onnx/models/w600k_mbf.onnx',
      sfaceModelUrl: 'https://static.tpsentinel.com/vendor/onnx/models/face_recognition_sface_2021dec_int8.onnx',
      onnxWasmBaseUrl: 'https://static.tpsentinel.com/vendor/onnx/runtime-web/',
    })
    return recognizerPromise
  }

  function revokeUrl(url: string | undefined) {
    if (url) URL.revokeObjectURL(url)
  }

  async function imageBitmapToUrl(bitmap: ImageBitmap): Promise<string> {
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d')

    if (!context) {
      throw new Error('Could not create preview canvas.')
    }

    context.drawImage(bitmap, 0, 0)
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) resolve(value)
        else reject(new Error('Could not create cropped preview blob.'))
      }, 'image/png')
    })

    return URL.createObjectURL(blob)
  }

  async function waitForVideoFrame(video: HTMLVideoElement): Promise<void> {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth && video.videoHeight) {
      return
    }

    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        cleanup()
        reject(new Error('Camera frame was not ready yet.'))
      }, 3000)
      const cleanup = () => {
        window.clearTimeout(timeout)
        video.removeEventListener('loadeddata', handleReady)
        video.removeEventListener('canplay', handleReady)
      }
      const handleReady = () => {
        cleanup()
        resolve()
      }

      video.addEventListener('loadeddata', handleReady, { once: true })
      video.addEventListener('canplay', handleReady, { once: true })
    })
  }

  async function captureVideoFrame(video: HTMLVideoElement, filename: string): Promise<File> {
    await waitForVideoFrame(video)

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')

    if (!context) {
      throw new Error('Could not create capture canvas.')
    }

    context.translate(canvas.width, 0)
    context.scale(-1, 1)
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) resolve(value)
        else reject(new Error('Could not capture webcam frame.'))
      }, 'image/png')
    })

    return new File([blob], filename, { type: 'image/png' })
  }

  function resetRecognitionOutput() {
    activeRunId = undefined
    setResult(undefined)
    setPipelineRows([])
    setReferenceFace(undefined)
    setProbeFace(undefined)
    setReferenceDimensions(undefined)
    setProbeDimensions(undefined)
    revokeUrl(referenceCropObjectUrl)
    revokeUrl(probeCropObjectUrl)
    referenceCropObjectUrl = undefined
    probeCropObjectUrl = undefined
    setReferenceCropPreview(undefined)
    setProbeCropPreview(undefined)
    setCropPreview(undefined)
  }

  async function stopContinuousRun() {
    if (!continuousRun) {
      return
    }

    await continuousRun.stop()
    continuousRun = undefined
    setIsContinuousRunning(false)
  }

  function handleReferenceUpload(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]

    if (!file) return

    revokeUrl(referenceObjectUrl)
    referenceObjectUrl = URL.createObjectURL(file)
    setReferenceFile(file)
    setReferencePreview(referenceObjectUrl)
    setReferenceMode('static')
    setReferenceSourceDetail('Uploaded image source')
    void stopContinuousRun()
    resetRecognitionOutput()
  }

  function handleProbeUpload(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]

    if (!file) return

    revokeUrl(probeObjectUrl)
    probeObjectUrl = URL.createObjectURL(file)
    setProbeFile(file)
    setProbePreview(probeObjectUrl)
    void stopContinuousRun()
    resetRecognitionOutput()
  }

  async function selectLiveMode() {
    setProbeMode('live')
    resetRecognitionOutput()

    if (!cameraStream()) {
      await startCamera()
    }
  }

  function selectStaticMode() {
    setProbeMode('static')
    void stopContinuousRun()
    resetRecognitionOutput()

    if (referenceMode() !== 'live') {
      stopCamera()
    }
  }

  async function selectReferenceLiveMode() {
    setReferenceMode('live')
    void stopContinuousRun()
    resetRecognitionOutput()

    if (!cameraStream()) {
      await startCamera()
    }
  }

  function selectReferenceStaticMode() {
    setReferenceMode('static')
    void stopContinuousRun()
    resetRecognitionOutput()

    if (probeMode() !== 'live') {
      stopCamera()
    }
  }

  async function captureReferenceFromCamera() {
    try {
      setCameraError(undefined)

      if (!cameraStream()) {
        await startCamera()
      }

      const video = referenceVideoRef ?? videoRef
      if (!video) {
        throw new Error('Reference camera preview is not ready.')
      }

      const file = await captureVideoFrame(video, `reference-capture-${Date.now()}.png`)
      revokeUrl(referenceObjectUrl)
      referenceObjectUrl = URL.createObjectURL(file)
      setReferenceFile(file)
      setReferencePreview(referenceObjectUrl)
      setReferenceMode('static')
      setReferenceSourceDetail('Captured webcam frame')
      void stopContinuousRun()
      resetRecognitionOutput()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not capture reference image.'
      setCameraError(message)
    }
  }

  async function startCamera() {
    try {
      setCameraError(undefined)
      setIsStartingCamera(true)

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      })

      setCameraStream(stream)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Camera access was denied or unavailable.'
      setCameraError(message)
      stopCamera()
    } finally {
      setIsStartingCamera(false)
    }
  }

  function stopCamera() {
    void stopContinuousRun()
    cameraStream()?.getTracks().forEach((track) => track.stop())
    setCameraStream(undefined)
  }

  function toggleCamera() {
    if (cameraStream()) {
      stopCamera()
      resetRecognitionOutput()
      return
    }

    void startCamera()
  }

  function selectRecognitionModel(model: FaceRecognitionModel) {
    if (recognitionModel() === model) return

    setRecognitionModel(model)
    setDistanceMetric('cosine')
    void stopContinuousRun()
    resetRecognitionOutput()
  }

  function selectDistanceMetric(metric: FaceRecognitionDistanceMetric) {
    if (distanceMetric() === metric) return

    setDistanceMetric(metric)
    void stopContinuousRun()
    resetRecognitionOutput()
  }

  function handlePipelineUpdate(update: FaceRecognitionPipelineUpdate) {
    if (activeRunId !== update.runId) {
      activeRunId = update.runId
      setPipelineRows((rows) =>
        update.branch === 'probe' ? rows.filter((row) => row.branch === 'reference') : [],
      )
    }

    setPipelineRows((rows) => [...rows, update])

    const sourceImage = update.data?.sourceImage
    if (sourceImage) {
      const dimensions = { width: sourceImage.width, height: sourceImage.height }
      if (update.branch === 'reference') setReferenceDimensions(dimensions)
      else setProbeDimensions(dimensions)
    }

    const face = update.data?.detectedFaces?.[0]
    if (face) {
      if (update.branch === 'reference') setReferenceFace(face)
      else setProbeFace(face)
    }

    const croppedFace = update.data?.croppedFace
    if (croppedFace) {
      void imageBitmapToUrl(croppedFace).then((url) => {
        if (update.branch === 'reference') {
          revokeUrl(referenceCropObjectUrl)
          referenceCropObjectUrl = url
          setReferenceCropPreview(url)
        } else {
          revokeUrl(probeCropObjectUrl)
          probeCropObjectUrl = url
          setProbeCropPreview(url)
        }
      })
    }
  }

  async function verifyCandidate() {
    const reference = referenceFile()

    if (!reference) {
      return
    }

    try {
      setCameraError(undefined)
      setIsVerifying(true)
      const recognizer = await getRecognizer()
      const probeSource = (() => {
        if (probeMode() === 'live') {
          if (!videoRef) return
          return { type: 'video' as const, video: videoRef }
        }

        const probe = probeFile()
        if (!probe) return
        return { type: 'image-file' as const, file: probe }
      })()

      if (!probeSource) {
        return
      }

      const nextResult = await recognizer.checkOnce({
        model: recognitionModel(),
        distanceMetric: distanceMetric(),
        reference: { type: 'image-file', file: reference },
        probe: probeSource,
        onUpdate: handlePipelineUpdate,
      })
      setResult(nextResult)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Face recognition failed.'
      setCameraError(message)
    } finally {
      setIsVerifying(false)
    }
  }

  async function startContinuousPipeline() {
    const reference = referenceFile()
    const video = videoRef

    if (!reference || !video || probeMode() !== 'live') {
      return
    }

    await stopContinuousRun()
    setCameraError(undefined)
    const recognizer = await getRecognizer()
    continuousRun = await recognizer.startContinuous({
      model: recognitionModel(),
      distanceMetric: distanceMetric(),
      reference: { type: 'image-file', file: reference },
      probe: { type: 'video', video },
      intervalMs: 1000,
      onUpdate: handlePipelineUpdate,
      onResult: setResult,
      onError: (error) => {
        setCameraError(error.message)
      },
    })
    setIsContinuousRunning(true)
  }

  onCleanup(() => {
    void stopContinuousRun()
    stopCamera()
    revokeUrl(referenceObjectUrl)
    revokeUrl(probeObjectUrl)
    revokeUrl(referenceCropObjectUrl)
    revokeUrl(probeCropObjectUrl)
  })

  return (
    <div class="min-h-svh bg-[#F8FAFC] text-slate-900 antialiased">
      <main class="mx-auto flex w-full max-w-[1440px] flex-col gap-3 px-3 py-3 sm:px-4">
        <section class="grid grid-cols-1 gap-3 lg:grid-cols-2" aria-label="Face comparison inputs">
          <ImagePanel
            branch="reference"
            title="Reference Face"
            subtitle="(Enrolled Model)"
            modeLabel={
              <div class="inline-flex rounded-md border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
                <button
                  class="rounded px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                  classList={{
                    'border border-slate-200/70 bg-white font-medium text-slate-900 shadow-sm':
                      referenceMode() === 'static',
                  }}
                  type="button"
                  onClick={selectReferenceStaticMode}
                >
                  Upload
                </button>
                <button
                  class="rounded px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                  classList={{
                    'border border-slate-200/70 bg-white font-medium text-slate-900 shadow-sm':
                      referenceMode() === 'live',
                  }}
                  type="button"
                  onClick={() => void selectReferenceLiveMode()}
                >
                  Camera
                </button>
              </div>
            }
            imageUrl={referenceMode() === 'static' ? referencePreview() : undefined}
            videoRef={(element) => {
              referenceVideoRef = element
              element.srcObject = cameraStream() ?? null
            }}
            showVideo={referenceMode() === 'live' && Boolean(cameraStream())}
            croppedUrl={referenceCropPreview()}
            face={referenceFace()}
            dimensions={referenceDimensions()}
            mirrored={referenceMode() === 'live'}
            onEmptyClick={() => {
              if (referenceMode() === 'live') {
                void startCamera()
                return
              }

              referenceInputRef?.click()
            }}
            emptyTitle={referenceMode() === 'live' ? 'Camera Capture Required' : 'No Reference Enrolled'}
            emptyDetail={
              referenceMode() === 'live'
                ? 'Start the camera and capture a reference frame.'
                : 'Upload or capture a biometric image to create the reference enrollment.'
            }
            metadataTitle={referenceCropPreview() ? 'Cropped reference ready' : 'Reference image required'}
            metadataDetail={referenceSourceDetail()}
            onPreviewCrop={(url) =>
              setCropPreview({
                title: 'Cropped Reference',
                detail: 'Reference enrollment crop',
                url,
              })
            }
            action={
              <Show
                when={referenceMode() === 'live'}
                fallback={
                  <label
                    class="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900"
                    for="reference-input"
                  >
                    <span class="material-symbols-outlined text-[15px]">file_upload</span>
                    <span>{referencePreview() ? 'Replace' : 'Upload'}</span>
                    <input
                      ref={referenceInputRef}
                      id="reference-input"
                      class="sr-only"
                      type="file"
                      accept="image/*"
                      onChange={handleReferenceUpload}
                    />
                  </label>
                }
              >
                <button
                  class="flex shrink-0 items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-[11px] font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
                  type="button"
                  disabled={isStartingCamera()}
                  onClick={() => {
                    if (cameraStream()) {
                      void captureReferenceFromCamera()
                      return
                    }

                    void startCamera()
                  }}
                >
                  <span class="material-symbols-outlined text-[16px]">
                    {cameraStream() ? 'photo_camera' : 'videocam'}
                  </span>
                  <span>
                    {isStartingCamera() ? 'Starting' : cameraStream() ? 'Capture' : 'Start Camera'}
                  </span>
                </button>
              </Show>
            }
          />

          <ImagePanel
            branch="probe"
            title="Probe Candidate"
            subtitle="(Live Evaluation)"
            modeLabel={
              <div class="inline-flex rounded-md border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
                <button
                  class="rounded px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                  classList={{
                    'border border-slate-200/70 bg-white font-medium text-slate-900 shadow-sm':
                      probeMode() === 'live',
                  }}
                  type="button"
                  onClick={() => void selectLiveMode()}
                >
                  Webcam Stream
                </button>
                <button
                  class="rounded px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                  classList={{
                    'border border-slate-200/70 bg-white font-medium text-slate-900 shadow-sm':
                      probeMode() === 'static',
                  }}
                  type="button"
                  onClick={selectStaticMode}
                >
                  Static Image
                </button>
              </div>
            }
            imageUrl={probeMode() === 'static' ? probePreview() : undefined}
            videoRef={(element) => {
              videoRef = element
              element.srcObject = cameraStream() ?? null
            }}
            showVideo={probeMode() === 'live' && Boolean(cameraStream())}
            croppedUrl={probeCropPreview()}
            face={probeFace()}
            dimensions={probeDimensions()}
            mirrored={probeMode() === 'live'}
            onEmptyClick={() => {
              if (probeMode() === 'live') {
                void startCamera()
                return
              }

              probeInputRef?.click()
            }}
            emptyTitle={probeMode() === 'live' ? 'Webcam Stream Required' : 'No Candidate Selected'}
            emptyDetail={
              probeMode() === 'live'
                ? 'Allow camera access to start live evaluation.'
                : 'Upload a static candidate image or switch to webcam stream.'
            }
            metadataTitle={probeCropPreview() ? 'Cropped probe ready' : 'Probe candidate required'}
            metadataDetail={probeMode() === 'live' ? 'Webcam frame stream' : 'Static probe image'}
            onPreviewCrop={(url) =>
              setCropPreview({
                title: 'Cropped Probe',
                detail: probeMode() === 'live' ? 'Latest webcam frame crop' : 'Static probe crop',
                url,
              })
            }
            action={
              <Show
                when={probeMode() === 'live'}
                fallback={
                  <label
                    class="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900"
                    for="probe-input"
                  >
                    <span class="material-symbols-outlined text-[15px]">file_upload</span>
                    <span>{probePreview() ? 'Replace' : 'Upload'}</span>
                    <input
                      ref={probeInputRef}
                      id="probe-input"
                      class="sr-only"
                      type="file"
                      accept="image/*"
                      onChange={handleProbeUpload}
                    />
                  </label>
                }
              >
                <button
                  class="flex shrink-0 items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-[11px] font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
                  type="button"
                  disabled={isStartingCamera()}
                  onClick={toggleCamera}
                >
                  <span class="material-symbols-outlined text-[16px]">
                    {cameraStream() ? 'videocam_off' : 'videocam'}
                  </span>
                  <span>
                    {isStartingCamera()
                      ? 'Starting'
                      : cameraStream()
                        ? 'Stop Stream'
                        : 'Start Stream'}
                  </span>
                </button>
              </Show>
            }
          />
        </section>

        <PipelineTimeline
          referenceRows={referenceRows()}
          probeRows={probeRows()}
          referenceTotalMs={referenceTotalMs()}
          probeTotalMs={probeTotalMs()}
          vectorLabel={vectorLabel()}
          compareLabel={compareLabel()}
        />

        <section class="rounded-lg border border-slate-200 bg-white p-3 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]">
          <div class="flex flex-col items-start justify-between gap-3 lg:flex-row lg:items-center">
            <div class="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div
                class="inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-semibold shadow-sm"
                classList={{
                  'border-emerald-200 bg-emerald-50 text-emerald-800': isReady() || Boolean(result()),
                  'border-amber-200 bg-amber-50 text-amber-800': !isReady() && !result() && !cameraError(),
                  'border-rose-200 bg-rose-50 text-rose-800': Boolean(cameraError()),
                }}
              >
                <span class="material-symbols-outlined text-[19px]">
                  {cameraError() ? 'error' : isReady() || result() ? 'check_circle' : 'pending'}
                </span>
                <span>{resultTitle()}</span>
              </div>
              <div>
                <p class="text-xs font-semibold text-slate-900">{resultDetail()}</p>
                <p class="mt-0.5 text-[11px] text-slate-500">
                  Operating threshold: {thresholdLabel()}. Embeddings are compared with {distanceMetric() === 'norm_l2' ? 'normalized L2 distance' : 'cosine similarity'}.
                </p>
              </div>
            </div>

            <div class="flex items-center gap-3 self-stretch rounded-lg border border-slate-200/80 bg-slate-50/80 px-3 py-1.5 sm:self-auto">
              <div class="flex flex-col">
                <span class="font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  {metricLabel()}
                </span>
                <div class="flex items-baseline gap-1">
                  <span class="font-mono text-2xl font-bold tracking-tight text-slate-900">{score()}</span>
                  <span class="font-mono text-xs text-slate-400">
                    {result()?.distanceMetric === 'norm_l2' ? 'distance' : '/ 1.000'}
                  </span>
                </div>
              </div>
              <div class="flex flex-col items-end justify-center border-l border-slate-200 pl-3">
                <span class="text-xs font-semibold text-emerald-700">{scorePercent()}</span>
                <span class="font-mono text-[10px] text-slate-400">score</span>
              </div>
            </div>
          </div>

          <div class="pt-3">
            <div class="flex flex-col justify-between gap-2 rounded-lg border border-slate-200/70 bg-slate-50 px-3 py-2 text-xs xl:flex-row xl:items-center">
              <div class="flex flex-wrap items-center gap-3 text-slate-600">
                <div class="flex items-center gap-2">
                  <span class="font-medium">Model:</span>
                  <div class="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
                    <button
                      class="rounded-md px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                      classList={{
                        'border border-slate-200/60 bg-white font-semibold text-slate-900 shadow-sm':
                          recognitionModel() === 'arcface-mbf',
                      }}
                      type="button"
                      onClick={() => selectRecognitionModel('arcface-mbf')}
                    >
                      ArcFace MBF
                    </button>
                    <button
                      class="rounded-md px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                      classList={{
                        'border border-slate-200/60 bg-white font-semibold text-slate-900 shadow-sm':
                          recognitionModel() === 'sface',
                      }}
                      type="button"
                      onClick={() => selectRecognitionModel('sface')}
                    >
                      SFace
                    </button>
                  </div>
                </div>

                <Show when={recognitionModel() === 'sface'}>
                  <div class="flex items-center gap-2">
                    <span class="font-medium">Metric:</span>
                    <div class="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
                      <button
                        class="rounded-md px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                        classList={{
                          'border border-slate-200/60 bg-white font-semibold text-slate-900 shadow-sm':
                            distanceMetric() === 'cosine',
                        }}
                        type="button"
                        onClick={() => selectDistanceMetric('cosine')}
                      >
                        Cosine
                      </button>
                      <button
                        class="rounded-md px-2.5 py-1 text-slate-600 transition-colors hover:text-slate-900"
                        classList={{
                          'border border-slate-200/60 bg-white font-semibold text-slate-900 shadow-sm':
                            distanceMetric() === 'norm_l2',
                        }}
                        type="button"
                        onClick={() => selectDistanceMetric('norm_l2')}
                      >
                        L2
                      </button>
                    </div>
                  </div>
                </Show>

                <div class="flex items-center gap-2">
                  <span class="font-medium">Threshold:</span>
                  <span class="rounded border border-slate-200 bg-white px-2 py-0.5 font-mono font-semibold text-slate-800 shadow-sm">
                    {thresholdLabel()}
                  </span>
                </div>
              </div>
              <Show
                when={probeMode() === 'live'}
                fallback={
                  <button
                    class="flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
                    type="button"
                    disabled={!isReady()}
                    onClick={() => void verifyCandidate()}
                  >
                    <span class="material-symbols-outlined text-[16px]">verified</span>
                    <span>{isVerifying() ? 'Verifying' : 'Verify Candidate'}</span>
                  </button>
                }
              >
                <button
                  class="flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-xs font-medium shadow-sm transition-all active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
                  classList={{
                    'bg-rose-50 text-rose-700 ring-1 ring-inset ring-rose-200 hover:bg-rose-100':
                      isContinuousRunning(),
                    'bg-slate-900 text-white hover:bg-slate-800': !isContinuousRunning(),
                  }}
                  type="button"
                  disabled={!canRunContinuous() && !isContinuousRunning()}
                  onClick={() => {
                    if (isContinuousRunning()) {
                      void stopContinuousRun()
                      return
                    }

                    void startContinuousPipeline()
                  }}
                >
                  <span class="material-symbols-outlined text-[16px]">
                    {isContinuousRunning() ? 'stop_circle' : 'play_circle'}
                  </span>
                  <span>{isContinuousRunning() ? 'Stop Continuous' : 'Run Continuous'}</span>
                </button>
              </Show>
            </div>
          </div>
        </section>
      </main>

      <Show when={cropPreview()}>
        {(preview) => (
          <div
            class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-sm"
            role="dialog"
            aria-modal="true"
            aria-labelledby="crop-preview-title"
            onClick={() => setCropPreview(undefined)}
          >
            <div
              class="w-full max-w-md rounded-xl border border-slate-200 bg-white shadow-2xl"
              onClick={(event) => event.stopPropagation()}
            >
              <div class="flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
                <div class="min-w-0">
                  <h2 id="crop-preview-title" class="truncate text-sm font-semibold text-slate-900">
                    {preview().title}
                  </h2>
                  <p class="mt-0.5 truncate font-mono text-xs text-slate-500">{preview().detail}</p>
                </div>
                <button
                  class="flex size-8 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-500 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900"
                  type="button"
                  title="Close preview"
                  onClick={() => setCropPreview(undefined)}
                >
                  <span class="material-symbols-outlined text-[18px]">close</span>
                </button>
              </div>
              <div class="bg-slate-100 p-5">
                <div class="mx-auto aspect-square w-full max-w-72 overflow-hidden rounded-lg border border-slate-300 bg-white shadow-inner">
                  <img class="size-full object-contain" src={preview().url} alt={preview().title} />
                </div>
              </div>
            </div>
          </div>
        )}
      </Show>
    </div>
  )
}

function ImagePanel(props: {
  branch: 'reference' | 'probe'
  title: string
  subtitle: string
  modeLabel: JSX.Element
  imageUrl?: string
  croppedUrl?: string
  face?: FaceDetectionSummary
  dimensions?: SourceDimensions
  mirrored?: boolean
  showVideo?: boolean
  videoRef?: (element: HTMLVideoElement) => void
  emptyTitle?: string
  emptyDetail?: string
  metadataTitle: string
  metadataDetail: string
  action: JSX.Element
  onEmptyClick: () => void
  onPreviewCrop?: (url: string) => void
}) {
  let mediaFrameRef: HTMLDivElement | undefined
  const [viewportDimensions, setViewportDimensions] = createSignal<SourceDimensions>()

  createEffect(() => {
    if (!mediaFrameRef) return

    const updateViewportDimensions = () => {
      setViewportDimensions({
        width: mediaFrameRef?.clientWidth ?? 0,
        height: mediaFrameRef?.clientHeight ?? 0,
      })
    }
    const observer = new ResizeObserver(updateViewportDimensions)

    updateViewportDimensions()
    observer.observe(mediaFrameRef)
    onCleanup(() => observer.disconnect())
  })

  return (
    <article class="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-3 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]">
      <div class="flex items-center justify-between gap-3">
        <div class="flex min-w-0 items-center gap-2">
          <span class="flex size-5 shrink-0 items-center justify-center rounded border border-slate-200 bg-slate-100 font-mono text-xs font-semibold text-slate-700">
            {props.branch === 'reference' ? '1' : '2'}
          </span>
          <h2 class="text-xs font-semibold tracking-tight text-slate-900 sm:text-sm">{props.title}</h2>
          <span class="hidden text-xs font-normal text-slate-500 sm:inline">{props.subtitle}</span>
        </div>
        {props.modeLabel}
      </div>

      <div
        ref={mediaFrameRef}
        class="relative flex aspect-video max-h-[42svh] w-full items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-50"
      >
        <Show when={props.imageUrl}>
          <img class="size-full object-cover" src={props.imageUrl} alt={`${props.title} source`} />
        </Show>
        <Show when={props.showVideo}>
          <>
            <video
              ref={(element) => props.videoRef?.(element)}
              class="size-full scale-x-[-1] object-cover"
              autoplay
              muted
              playsinline
            />
            <div class="pointer-events-none absolute bottom-3 left-3">
              <span class="inline-flex items-center gap-1.5 rounded border border-slate-200 bg-white/90 px-2 py-0.5 font-mono text-[11px] text-slate-600 shadow-sm backdrop-blur">
                <span class="size-1.5 rounded-full bg-emerald-500"></span>
                30.0 FPS
              </span>
            </div>
          </>
        </Show>
        <Show when={props.face && props.dimensions}>
          <FaceOverlay
            face={props.face}
            dimensions={props.dimensions}
            viewportDimensions={viewportDimensions()}
            mirrored={props.mirrored}
          />
        </Show>
        <Show when={!props.imageUrl && !props.showVideo}>
          <button
            class="absolute inset-0 flex cursor-pointer flex-col items-center justify-center bg-white/95 p-6 text-center transition-colors hover:bg-slate-50"
            type="button"
            onClick={props.onEmptyClick}
          >
            <div class="mb-2 flex size-10 items-center justify-center rounded-full border border-slate-100 bg-slate-100 text-slate-500">
              <span class="material-symbols-outlined text-xl">
                {props.branch === 'reference' ? 'image' : 'face_retouching_off'}
              </span>
            </div>
            <h3 class="text-sm font-semibold text-slate-900">
              {props.emptyTitle ?? 'No Reference Enrolled'}
            </h3>
            <p class="mt-1 max-w-xs text-xs text-slate-500">
              {props.emptyDetail ?? 'Upload a biometric image to create the reference enrollment.'}
            </p>
          </button>
        </Show>
      </div>

      <div class="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 p-2">
        <div class="flex min-w-0 items-center gap-2">
          <button
            class="group relative flex size-9 shrink-0 items-center justify-center overflow-hidden rounded border border-slate-300 bg-white shadow-sm transition-colors disabled:cursor-default"
            type="button"
            disabled={!props.croppedUrl}
            title={props.croppedUrl ? 'Preview crop' : undefined}
            onClick={() => {
              if (props.croppedUrl) props.onPreviewCrop?.(props.croppedUrl)
            }}
          >
            <Show
              when={props.croppedUrl}
              fallback={<span class="material-symbols-outlined text-[18px] text-slate-300">center_focus_strong</span>}
            >
              <img class="size-full object-cover" src={props.croppedUrl} alt={`${props.title} cropped face`} />
              <span class="absolute inset-0 flex items-center justify-center bg-slate-950/0 text-white opacity-0 transition-all group-hover:bg-slate-950/35 group-hover:opacity-100">
                <span class="material-symbols-outlined text-[16px]">zoom_in</span>
              </span>
            </Show>
          </button>
          <div class="min-w-0">
            <p class="truncate text-[11px] font-semibold text-slate-800">{props.metadataTitle}</p>
            <p class="mt-0.5 truncate font-mono text-[10px] text-slate-500">{props.metadataDetail}</p>
          </div>
        </div>
        {props.action}
      </div>
    </article>
  )
}

function FaceOverlay(props: {
  face?: FaceDetectionSummary
  dimensions?: SourceDimensions
  viewportDimensions?: SourceDimensions
  mirrored?: boolean
}) {
  const mapSourcePoint = (point: PointLike) => {
    const source = props.dimensions
    const viewport = props.viewportDimensions

    if (!source || !viewport?.width || !viewport.height) {
      return {
        x: `${(props.mirrored ? 1 - point.x : point.x) * 100}%`,
        y: `${point.y * 100}%`,
      }
    }

    const scale = Math.max(viewport.width / source.width, viewport.height / source.height)
    const renderedWidth = source.width * scale
    const renderedHeight = source.height * scale
    const offsetX = (viewport.width - renderedWidth) / 2
    const offsetY = (viewport.height - renderedHeight) / 2
    const sourceX = props.mirrored ? (1 - point.x) * source.width : point.x * source.width
    const sourceY = point.y * source.height

    return {
      x: `${offsetX + sourceX * scale}px`,
      y: `${offsetY + sourceY * scale}px`,
    }
  }

  const boxStyle = createMemo(() => {
    const face = props.face
    const dimensions = props.dimensions
    const viewport = props.viewportDimensions

    if (!face || !dimensions) return {}

    if (!viewport?.width || !viewport.height) {
      const left = props.mirrored
        ? 100 - ((face.box.x + face.box.width) / dimensions.width) * 100
        : (face.box.x / dimensions.width) * 100

      return {
        left: `${left}%`,
        top: `${(face.box.y / dimensions.height) * 100}%`,
        width: `${(face.box.width / dimensions.width) * 100}%`,
        height: `${(face.box.height / dimensions.height) * 100}%`,
      }
    }

    const scale = Math.max(viewport.width / dimensions.width, viewport.height / dimensions.height)
    const renderedWidth = dimensions.width * scale
    const renderedHeight = dimensions.height * scale
    const offsetX = (viewport.width - renderedWidth) / 2
    const offsetY = (viewport.height - renderedHeight) / 2
    const sourceLeft = props.mirrored
      ? dimensions.width - face.box.x - face.box.width
      : face.box.x

    return {
      left: `${offsetX + sourceLeft * scale}px`,
      top: `${offsetY + face.box.y * scale}px`,
      width: `${face.box.width * scale}px`,
      height: `${face.box.height * scale}px`,
    }
  })

  return (
    <>
      <div class="pointer-events-none absolute rounded-md border border-slate-900/70 bg-slate-900/5 shadow-sm" style={boxStyle()}>
        <span class="absolute -top-2.5 left-2 rounded bg-slate-900 px-1.5 py-0.5 font-mono text-[10px] text-white shadow-sm">
          {props.face ? `${Math.round(props.face.confidence * 100)}%` : ''}
        </span>
      </div>
      <For each={props.face?.keypoints ?? []}>
        {(keypoint) => (
          <span
            class="pointer-events-none absolute size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-950 shadow-[0_0_0_2px_rgba(255,255,255,0.85)]"
            style={{
              left: mapSourcePoint(keypoint).x,
              top: mapSourcePoint(keypoint).y,
            }}
            title={keypoint.label}
          />
        )}
      </For>
      <div class="pointer-events-none absolute left-3 top-3">
        <span class="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1 text-xs font-medium text-slate-800 shadow-sm backdrop-blur">
          <span class="size-2 rounded-full bg-emerald-500"></span>
          1 face detected
        </span>
      </div>
    </>
  )
}

function PipelineTimeline(props: {
  referenceRows: PipelineRow[]
  probeRows: PipelineRow[]
  referenceTotalMs: number
  probeTotalMs: number
  vectorLabel: string
  compareLabel: string
}) {
  return (
    <section class="rounded-lg border border-slate-200 bg-white p-3 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-[18px] text-slate-500">alt_route</span>
          <h3 class="text-xs font-semibold tracking-tight text-slate-900 sm:text-sm">
            Execution Trace & Inference Pipelines
          </h3>
          <span class="hidden text-[10px] text-slate-400 sm:inline">
            Per-frame SIMD worker latency
          </span>
        </div>
        <span class="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 font-mono text-[10px] text-slate-500">
          WASM SIMD Parallel
        </span>
      </div>

      <div class="flex flex-col gap-2 pt-2">
        <PipelineBranch
          label="Reference Stream"
          rows={props.referenceRows}
          totalMs={props.referenceTotalMs}
          emptyStages={['Image Ingest', 'Detection', 'Align & Crop', props.vectorLabel]}
          vectorLabel={props.vectorLabel}
        />
        <PipelineBranch
          label="Live Candidate Stream"
          rows={props.probeRows}
          totalMs={props.probeTotalMs}
          emptyStages={['Frame Ingest', 'Detection', 'Align & Crop', props.vectorLabel, props.compareLabel]}
          vectorLabel={props.vectorLabel}
          compareLabel={props.compareLabel}
        />
      </div>
    </section>
  )
}

function PipelineBranch(props: {
  label: string
  rows: PipelineRow[]
  totalMs: number
  emptyStages: string[]
  vectorLabel?: string
  compareLabel?: string
}) {
  const tiles = createMemo(() =>
    compactPipelineRows(
      props.rows,
      props.emptyStages,
      props.vectorLabel ?? '512-d Vector',
      props.compareLabel ?? 'Cosine Compare',
    ),
  )

  return (
    <div class="flex flex-col gap-1">
      <div class="flex items-center justify-between text-[11px] font-medium text-slate-500">
        <span class="flex items-center gap-1.5">
          <span class="size-1.5 rounded-full bg-slate-400"></span>
          {props.label}
        </span>
        <span class="font-mono text-[11px] text-slate-400">
          Cumulative: {props.totalMs.toFixed(1)} ms
        </span>
      </div>
      <div class="grid grid-cols-1 gap-1.5 md:grid-cols-2 xl:grid-cols-5">
        <For each={tiles()}>
          {(tile) => (
            <div
              class="flex items-center justify-between gap-2 rounded-md border bg-slate-50 px-2 py-1.5"
              classList={{
                'border-slate-200': tile.status === 'idle',
                'border-emerald-200': tile.status === 'success',
                'border-amber-200': tile.status === 'running' || tile.status === 'warning',
                'border-rose-200': tile.status === 'error',
              }}
            >
              <div class="flex min-w-0 items-center gap-1.5">
                <span
                  class="material-symbols-outlined shrink-0 text-[15px]"
                  classList={{
                    'text-slate-400': tile.status === 'idle',
                    'text-emerald-600': tile.status === 'success',
                    'text-amber-600': tile.status === 'running' || tile.status === 'warning',
                    'text-rose-600': tile.status === 'error',
                  }}
                >
                  {tile.status === 'error'
                    ? 'error'
                    : tile.status === 'running'
                      ? 'progress_activity'
                      : tile.status === 'idle'
                        ? 'radio_button_unchecked'
                        : 'check_circle'}
                </span>
                <span class="truncate text-[11px] font-medium text-slate-700">{tile.label}</span>
              </div>
              <span class="shrink-0 font-mono text-[10px] text-slate-500">
                {tile.durationMs === undefined ? '--' : `${tile.durationMs.toFixed(1)} ms`}
              </span>
            </div>
          )}
        </For>
      </div>
    </div>
  )
}

function compactPipelineRows(
  rows: PipelineRow[],
  emptyStages: string[],
  vectorLabel: string,
  compareLabel: string,
) {
  if (!rows.length) {
    return emptyStages.map((label) => ({
      label,
      status: 'idle' as const,
      durationMs: undefined,
    }))
  }

  const labels = emptyStages.map((label) => ({
    label,
    status: 'idle' as PipelineRow['status'],
    durationMs: undefined as number | undefined,
  }))

  for (const row of rows) {
    const label =
      row.stage === 'embedding-running' || row.stage === 'embedding-ready'
        ? vectorLabel
        : row.stage === 'comparing' || row.stage === 'completed'
          ? compareLabel
        : row.stage === 'source-ready' && row.branch === 'probe'
          ? 'Frame Ingest'
          : STAGE_LABELS[row.stage]
    const target = labels.find((tile) => tile.label === label)

    if (!target) continue

    target.status = row.status
    if (row.durationMs !== undefined) {
      target.durationMs = row.durationMs
    }
  }

  return labels
}

function totalDuration(rows: PipelineRow[]): number {
  return rows.reduce((total, row) => total + (row.durationMs ?? 0), 0)
}
