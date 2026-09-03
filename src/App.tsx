import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

type ProbeMode = 'static' | 'live'

export function App() {
  let referenceInputRef: HTMLInputElement | undefined
  let probeInputRef: HTMLInputElement | undefined
  let videoRef: HTMLVideoElement | undefined
  let referenceObjectUrl: string | undefined
  let probeObjectUrl: string | undefined

  const [referencePreview, setReferencePreview] = createSignal<string>()
  const [probePreview, setProbePreview] = createSignal<string>()
  const [probeMode, setProbeMode] = createSignal<ProbeMode>('static')
  const [cameraStream, setCameraStream] = createSignal<MediaStream>()
  const [cameraError, setCameraError] = createSignal<string>()
  const [isStartingCamera, setIsStartingCamera] = createSignal(false)
  const [verificationScore, setVerificationScore] = createSignal<string>()

  const hasProbe = createMemo(() =>
    probeMode() === 'live' ? Boolean(cameraStream()) : Boolean(probePreview()),
  )
  const isReady = createMemo(() => Boolean(referencePreview()) && hasProbe())

  const pipelineStatus = createMemo(() => {
    if (cameraError()) return 'Camera unavailable'
    if (verificationScore()) return 'Verification complete'
    if (isReady()) return 'Pipeline Active'
    if (!referencePreview()) return hasProbe() ? 'Reference Required' : 'Awaiting Inputs'
    return 'Probe Required'
  })

  const resultTitle = createMemo(() => {
    if (cameraError()) return 'CAMERA UNAVAILABLE'
    if (verificationScore()) return 'VERIFICATION SIMULATED'
    if (isReady()) return 'READY TO VERIFY'
    if (!referencePreview() && !hasProbe()) return 'AWAITING INPUTS'
    if (!referencePreview()) return 'REFERENCE REQUIRED'
    return 'PROBE REQUIRED'
  })

  const resultDetail = createMemo(() => {
    if (cameraError()) return cameraError()
    if (verificationScore()) {
      return `Similarity score ${verificationScore()}. Recognition model integration can be added next.`
    }
    if (isReady()) {
      return probeMode() === 'live'
        ? 'Live webcam frames are available for probe evaluation.'
        : 'The uploaded probe image is available for static evaluation.'
    }
    if (!referencePreview() && !hasProbe()) {
      return 'Upload a reference image, then provide a probe candidate.'
    }
    if (!referencePreview()) return 'Choose a reference image before verification.'
    return probeMode() === 'live'
      ? 'Allow webcam access to start live probe evaluation.'
      : 'Choose a static probe image or switch to live stream.'
  })

  const score = createMemo(() => verificationScore() ?? '0.000')
  const scorePercent = createMemo(() => `${Math.round(Number(score()) * 1000) / 10}%`)

  createEffect(() => {
    const stream = cameraStream()

    if (videoRef) {
      videoRef.srcObject = stream ?? null
    }
  })

  function revokeUrl(url: string | undefined) {
    if (url) URL.revokeObjectURL(url)
  }

  function handleReferenceUpload(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]

    if (!file) return

    revokeUrl(referenceObjectUrl)
    referenceObjectUrl = URL.createObjectURL(file)
    setReferencePreview(referenceObjectUrl)
    setVerificationScore(undefined)
  }

  function handleProbeUpload(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]

    if (!file) return

    revokeUrl(probeObjectUrl)
    probeObjectUrl = URL.createObjectURL(file)
    setProbePreview(probeObjectUrl)
    setVerificationScore(undefined)
  }

  async function selectLiveMode() {
    setProbeMode('live')
    setVerificationScore(undefined)

    if (!cameraStream()) {
      await startCamera()
    }
  }

  function selectStaticMode() {
    setProbeMode('static')
    setVerificationScore(undefined)
    stopCamera()
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
    cameraStream()?.getTracks().forEach((track) => track.stop())
    setCameraStream(undefined)
  }

  function toggleCamera() {
    if (cameraStream()) {
      stopCamera()
      return
    }

    void startCamera()
  }

  function resetWorkbench() {
    setVerificationScore(undefined)
    setCameraError(undefined)
  }

  function verifyCandidate() {
    setVerificationScore(probeMode() === 'live' ? '0.781' : '0.736')
  }

  onCleanup(() => {
    stopCamera()
    revokeUrl(referenceObjectUrl)
    revokeUrl(probeObjectUrl)
  })

  return (
    <div class="min-h-svh bg-[#F8FAFC] text-slate-900 antialiased">
      <header class="sticky top-0 z-40 border-b border-slate-200/90 bg-white/95 shadow-[0_1px_2px_rgba(15,23,42,0.04)] backdrop-blur">
        <div class="mx-auto flex h-14 max-w-[1440px] items-center justify-between gap-4 px-4 sm:px-6">
          <div class="flex min-w-0 items-center gap-2.5">
            <div class="flex size-7 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white shadow-sm">
              <span class="material-symbols-outlined text-[17px]">fingerprint</span>
            </div>
            <div class="min-w-0">
              <p class="truncate text-xs font-semibold text-slate-900">Face Recognition POC</p>
              <p class="hidden text-[11px] font-medium text-slate-500 sm:block">
                Reference enrollment and probe evaluation
              </p>
            </div>
            <span class="ml-1 rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 font-mono text-[11px] font-medium text-slate-600">
              v0.1.0
            </span>
          </div>

          <div class="hidden items-center gap-3 md:flex">
            <div class="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
              <button
                class="rounded-md border border-slate-200/60 bg-white px-3 py-1 font-semibold text-slate-900 shadow-sm"
                type="button"
              >
                ArcFace MBF
              </button>
              <button class="rounded-md px-3 py-1 text-slate-600 transition-colors hover:text-slate-900" type="button">
                SFace v2
              </button>
            </div>
            <span class="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 font-mono text-xs text-slate-600 shadow-sm">
              <span class="size-1.5 rounded-full bg-emerald-500"></span>
              WASM SIMD
            </span>
          </div>

          <div class="flex items-center gap-2 sm:gap-3">
            <div class="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
              <span class="size-1.5 rounded-full bg-emerald-500"></span>
              <span>{pipelineStatus()}</span>
            </div>
            <div class="hidden h-4 w-px bg-slate-200 sm:block"></div>
            <button
              class="rounded-md border border-transparent p-1.5 text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-100 hover:text-slate-900"
              title="Reset evaluation"
              type="button"
              onClick={resetWorkbench}
            >
              <span class="material-symbols-outlined text-[19px]">refresh</span>
            </button>
          </div>
        </div>
      </header>

      <main class="mx-auto flex w-full max-w-[1440px] flex-col gap-6 px-4 py-6 sm:px-6">
        <section class="grid grid-cols-1 gap-6 lg:grid-cols-2" aria-label="Face comparison inputs">
          <article class="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]">
            <div class="flex items-center justify-between gap-3 border-b border-slate-100 pb-1">
              <div class="flex min-w-0 items-center gap-2">
                <span class="flex size-5 shrink-0 items-center justify-center rounded border border-slate-200 bg-slate-100 font-mono text-xs font-semibold text-slate-700">
                  1
                </span>
                <h2 class="text-sm font-semibold tracking-tight text-slate-900">Reference Face</h2>
                <span class="hidden text-xs font-normal text-slate-500 sm:inline">(Enrolled Model)</span>
              </div>
              <div class="inline-flex rounded-md border border-slate-200 bg-slate-100 p-0.5 text-xs font-medium">
                <span class="rounded border border-slate-200/70 bg-white px-2.5 py-1 font-medium text-slate-900 shadow-sm">
                  Static Image
                </span>
              </div>
            </div>

            <div class="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
              <Show
                when={referencePreview()}
                fallback={
                  <button
                    class="absolute inset-0 flex cursor-pointer flex-col items-center justify-center bg-white/95 p-6 text-center transition-colors hover:bg-slate-50"
                    type="button"
                    onClick={() => referenceInputRef?.click()}
                  >
                    <div class="mb-2 flex size-10 items-center justify-center rounded-full bg-slate-100 text-slate-500">
                      <span class="material-symbols-outlined text-xl">image</span>
                    </div>
                    <h3 class="text-sm font-semibold text-slate-900">No Reference Enrolled</h3>
                    <p class="mt-1 max-w-xs text-xs text-slate-500">
                      Upload a biometric image to create the reference enrollment.
                    </p>
                  </button>
                }
              >
                {(src) => (
                  <>
                    <img class="size-full object-cover" src={src()} alt="Uploaded reference face" />
                    <div class="pointer-events-none absolute inset-0 flex items-center justify-center">
                      <div class="relative h-56 w-48 rounded-md border border-slate-900/60 bg-slate-900/5 shadow-sm">
                        <span class="absolute -top-2.5 left-2.5 rounded bg-slate-900 px-1.5 py-0.5 font-mono text-[10px] text-white shadow-sm">
                          Ref ID: #00
                        </span>
                      </div>
                    </div>
                    <div class="pointer-events-none absolute left-3 top-3">
                      <span class="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1 text-xs font-medium text-slate-800 shadow-sm backdrop-blur">
                        <span class="size-2 rounded-full bg-emerald-500"></span>
                        Reference loaded
                      </span>
                    </div>
                  </>
                )}
              </Show>
            </div>

            <div class="flex items-center justify-between gap-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div class="min-w-0">
                <p class="text-xs font-semibold text-slate-800">
                  {referencePreview() ? 'Reference image ready' : 'Reference image required'}
                </p>
                <p class="mt-0.5 font-mono text-xs text-slate-500">Upload-only enrollment source</p>
              </div>
              <label
                class="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900"
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
            </div>
          </article>

          <article class="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]">
            <div class="flex items-center justify-between gap-3 border-b border-slate-100 pb-1">
              <div class="flex min-w-0 items-center gap-2">
                <span class="flex size-5 shrink-0 items-center justify-center rounded border border-slate-200 bg-slate-100 font-mono text-xs font-semibold text-slate-700">
                  2
                </span>
                <h2 class="text-sm font-semibold tracking-tight text-slate-900">Probe Candidate</h2>
                <span class="hidden text-xs font-normal text-slate-500 sm:inline">(Live Evaluation)</span>
              </div>
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
            </div>

            <div class="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
              <Show when={probeMode() === 'static' && probePreview()}>
                <img class="size-full object-cover" src={probePreview()} alt="Uploaded probe candidate" />
              </Show>

              <Show when={probeMode() === 'live' && cameraStream()}>
                <>
                  <video
                    ref={videoRef}
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

              <Show when={hasProbe()}>
                <div class="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <div class="relative h-56 w-48 rounded-md border border-slate-900/60 bg-slate-900/5 shadow-sm">
                    <span class="absolute -top-2.5 left-2.5 rounded bg-slate-900 px-1.5 py-0.5 font-mono text-[10px] text-white shadow-sm">
                      Candidate #0
                    </span>
                    <span class="absolute left-[28%] top-[35%] size-1.5 rounded-full bg-slate-900"></span>
                    <span class="absolute right-[28%] top-[35%] size-1.5 rounded-full bg-slate-900"></span>
                    <span class="absolute left-[48%] top-[52%] size-1.5 rounded-full bg-slate-900"></span>
                    <span class="absolute left-[32%] top-[68%] size-1.5 rounded-full bg-slate-900"></span>
                    <span class="absolute right-[32%] top-[68%] size-1.5 rounded-full bg-slate-900"></span>
                  </div>
                </div>
                <div class="pointer-events-none absolute left-3 top-3 flex items-center gap-2">
                  <span class="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1 text-xs font-medium text-slate-800 shadow-sm backdrop-blur">
                    <span class="size-2 rounded-full bg-emerald-500"></span>
                    Candidate source ready
                  </span>
                </div>
              </Show>

              <Show when={!hasProbe()}>
                <button
                  class="absolute inset-0 flex cursor-pointer flex-col items-center justify-center bg-white/95 p-6 text-center transition-colors hover:bg-slate-50 disabled:cursor-not-allowed"
                  type="button"
                  disabled={isStartingCamera()}
                  onClick={() => {
                    if (probeMode() === 'live') {
                      void startCamera()
                      return
                    }

                    probeInputRef?.click()
                  }}
                >
                  <div class="mb-2 flex size-10 items-center justify-center rounded-full border border-rose-100 bg-rose-50 text-rose-600">
                    <span class="material-symbols-outlined text-xl">
                      {probeMode() === 'live' ? 'videocam' : 'face_retouching_off'}
                    </span>
                  </div>
                  <h3 class="text-sm font-semibold text-slate-900">
                    {probeMode() === 'live' ? 'Webcam Stream Required' : 'No Candidate Selected'}
                  </h3>
                  <p class="mt-1 max-w-xs text-xs text-slate-500">
                    {probeMode() === 'live'
                      ? 'Allow camera access to start live evaluation.'
                      : 'Upload a static candidate image or switch to webcam stream.'}
                  </p>
                </button>
              </Show>
            </div>

            <div class="flex items-center justify-between gap-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div class="min-w-0">
                <p class="text-xs font-semibold text-slate-800">
                  {hasProbe() ? 'Probe candidate ready' : 'Probe candidate required'}
                </p>
                <p class="mt-0.5 font-mono text-xs text-slate-500">
                  {probeMode() === 'live' ? 'Webcam frame stream' : 'Static probe image'}
                </p>
              </div>

              <Show
                when={probeMode() === 'live'}
                fallback={
                  <label
                    class="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900"
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
                  class="flex shrink-0 items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
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
            </div>
          </article>
        </section>

        <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)] sm:p-6">
          <div class="flex flex-col items-start justify-between gap-6 border-b border-slate-100 pb-5 lg:flex-row lg:items-center">
            <div class="flex flex-col gap-4 sm:flex-row sm:items-center">
              <div
                class="inline-flex items-center gap-2 rounded-lg border px-3.5 py-2 text-sm font-semibold shadow-sm"
                classList={{
                  'border-emerald-200 bg-emerald-50 text-emerald-800': isReady() || Boolean(verificationScore()),
                  'border-amber-200 bg-amber-50 text-amber-800': !isReady() && !verificationScore() && !cameraError(),
                  'border-rose-200 bg-rose-50 text-rose-800': Boolean(cameraError()),
                }}
              >
                <span class="material-symbols-outlined text-[19px]">
                  {cameraError() ? 'error' : isReady() || verificationScore() ? 'check_circle' : 'pending'}
                </span>
                <span>{resultTitle()}</span>
              </div>
              <div>
                <p class="text-sm font-semibold text-slate-900">{resultDetail()}</p>
                <p class="mt-0.5 text-xs text-slate-500">
                  Operating threshold: 0.650. This POC currently simulates the comparison score.
                </p>
              </div>
            </div>

            <div class="flex items-center gap-4 self-stretch rounded-lg border border-slate-200/80 bg-slate-50/80 px-4 py-2.5 sm:self-auto">
              <div class="flex flex-col">
                <span class="font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  Similarity Metric
                </span>
                <div class="flex items-baseline gap-1">
                  <span class="font-mono text-3xl font-bold tracking-tight text-slate-900">{score()}</span>
                  <span class="font-mono text-xs text-slate-400">/ 1.000</span>
                </div>
              </div>
              <div class="flex flex-col items-end justify-center border-l border-slate-200 pl-4">
                <span class="text-xs font-semibold text-emerald-700">{scorePercent()}</span>
                <span class="font-mono text-[10px] text-slate-400">score</span>
              </div>
            </div>
          </div>

          <div class="flex flex-col gap-3 pt-5">
            <div class="flex items-center justify-between text-xs text-slate-500">
              <span class="font-medium text-slate-700">Classification Spectrum</span>
              <div class="hidden items-center gap-4 text-[11px] sm:flex">
                <span class="inline-flex items-center gap-1.5">
                  <span class="size-2 rounded-full bg-rose-500"></span>
                  Mismatch
                </span>
                <span class="inline-flex items-center gap-1.5">
                  <span class="size-2 rounded-full bg-amber-500"></span>
                  Review Zone
                </span>
                <span class="inline-flex items-center gap-1.5">
                  <span class="size-2 rounded-full bg-emerald-600"></span>
                  Match
                </span>
              </div>
            </div>

            <div class="relative flex h-2.5 w-full overflow-hidden rounded-full border border-slate-200 bg-slate-100">
              <div class="h-full w-[55%] border-r border-rose-200 bg-rose-100"></div>
              <div class="h-full w-[10%] border-r border-amber-200 bg-amber-100"></div>
              <div class="h-full w-[35%] bg-emerald-100"></div>
              <div
                class="absolute bottom-0 top-0 w-1 -translate-x-1/2 bg-slate-900 shadow-sm transition-all duration-300"
                style={{ left: verificationScore() ? scorePercent() : '0%' }}
              >
                <div class="absolute -left-[2px] -top-0.5 size-2 rounded-full bg-slate-900"></div>
              </div>
            </div>

            <div class="mt-1 flex flex-col justify-between gap-3 rounded-lg border border-slate-200/70 bg-slate-50 px-3.5 py-2 text-xs sm:flex-row sm:items-center">
              <div class="flex items-center gap-2 text-slate-600">
                <span class="font-medium">Verification Threshold:</span>
                <span class="rounded border border-slate-200 bg-white px-2 py-0.5 font-mono font-semibold text-slate-800 shadow-sm">
                  0.650
                </span>
              </div>
              <button
                class="flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
                type="button"
                disabled={!isReady()}
                onClick={verifyCandidate}
              >
                <span class="material-symbols-outlined text-[16px]">verified</span>
                <span>Verify Candidate</span>
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
