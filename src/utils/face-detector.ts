import type {
  FaceLandmarker as MediaPipeFaceLandmarker,
  FaceLandmarkerResult,
} from '@mediapipe/tasks-vision'

const MEDIAPIPE_WASM_BASE_URL = 'https://static.tpsentinel.com/vendor/mediapipe/wasm'
const MEDIAPIPE_FACE_LANDMARKER_MODEL_URL =
  'https://static.tpsentinel.com/vendor/mediapipe/models/face_landmarker.task'

const defaultVisionTasksLoader = () => import('@mediapipe/tasks-vision')

let visionTasksLoader = defaultVisionTasksLoader

export const loadVisionTasks = () => visionTasksLoader()

type VisionTasksModule = Awaited<ReturnType<typeof loadVisionTasks>>
type VisionFaceLandmarkerOptions = Parameters<
  VisionTasksModule['FaceLandmarker']['createFromOptions']
>[1]

export const setVisionTasksLoader = (loader?: typeof defaultVisionTasksLoader): void => {
  visionTasksLoader = loader ?? defaultVisionTasksLoader
}

export class FaceDetector {
  private landmarker: MediaPipeFaceLandmarker | null = null

  async init({ useModuleLoader = false }: { useModuleLoader?: boolean } = {}): Promise<void> {
    if (this.landmarker) {
      return
    }

    const vision = await loadVisionTasks()
    const { FilesetResolver, FaceLandmarker: MPFaceLandmarkerClass } = vision

    const wasmFileset = await FilesetResolver.forVisionTasks(
      MEDIAPIPE_WASM_BASE_URL,
      useModuleLoader,
    )

    this.landmarker = await this.createLandmarker(MPFaceLandmarkerClass, wasmFileset)
  }

  detectLandmarks(image: ImageBitmap | HTMLCanvasElement): FaceLandmarkerResult {
    if (!this.landmarker) {
      throw new Error('FaceLandmarker is not initialized. Call init() first.')
    }

    return this.landmarker.detect(image)
  }

  close(): void {
    if (this.landmarker) {
      try {
        this.landmarker.close()
      } catch (_err) {
        // MediaPipe close can throw when called after runtime teardown.
      }
      this.landmarker = null
    }
  }

  private async createLandmarker(
    landmarkerClass: VisionTasksModule['FaceLandmarker'],
    wasmFileset: Awaited<ReturnType<VisionTasksModule['FilesetResolver']['forVisionTasks']>>,
  ): Promise<MediaPipeFaceLandmarker> {
    return landmarkerClass.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: MEDIAPIPE_FACE_LANDMARKER_MODEL_URL,
        delegate: 'CPU',
      },
      numFaces: 1,
      minFaceDetectionConfidence: 0.75,
      minFacePresenceConfidence: 0.75,
      minTrackingConfidence: 0.75,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
      runningMode: 'IMAGE',
    } satisfies VisionFaceLandmarkerOptions)
  }
}
