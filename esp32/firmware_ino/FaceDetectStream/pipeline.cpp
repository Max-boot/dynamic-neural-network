// ============================================================================
//  pipeline.cpp  -  Shared state hub + inference task (runs on core 1).
// ============================================================================
#include "pipeline.h"
#include "config.h"
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

// ---- shared buffers --------------------------------------------------------
static float*   g_scene = nullptr;        // latest gray scene [SCENE*SCENE]
static bool     g_scene_ready = false;
static SemaphoreHandle_t g_scene_mtx = nullptr;

static Detection g_dets[MAX_DETS];
static int       g_ndets = 0;
static SemaphoreHandle_t g_dets_mtx = nullptr;

#define JPEG_CAP (48 * 1024)              // generous ceiling for one frame
static uint8_t* g_jpeg = nullptr;
static size_t   g_jpeg_len = 0;
static SemaphoreHandle_t g_jpeg_mtx = nullptr;

static float* palloc(size_t n) {
  float* p = (float*)heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!p) p = (float*)malloc(n * sizeof(float));
  return p;
}

bool shared_begin() {
  g_scene = palloc(SCENE * SCENE);
  g_jpeg  = (uint8_t*)heap_caps_malloc(JPEG_CAP, MALLOC_CAP_SPIRAM);
  if (!g_jpeg) g_jpeg = (uint8_t*)malloc(JPEG_CAP);
  g_scene_mtx = xSemaphoreCreateMutex();
  g_dets_mtx  = xSemaphoreCreateMutex();
  g_jpeg_mtx  = xSemaphoreCreateMutex();
  bool ok = g_scene && g_jpeg && g_scene_mtx && g_dets_mtx && g_jpeg_mtx;
  if (!ok) Serial.println("[shared] init failed");
  return ok;
}

// ---- scene -----------------------------------------------------------------
void shared_set_scene(const float* scene) {
  if (xSemaphoreTake(g_scene_mtx, portMAX_DELAY) == pdTRUE) {
    memcpy(g_scene, scene, SCENE * SCENE * sizeof(float));
    g_scene_ready = true;
    xSemaphoreGive(g_scene_mtx);
  }
}

bool shared_get_scene(float* dst) {
  bool ok = false;
  if (xSemaphoreTake(g_scene_mtx, portMAX_DELAY) == pdTRUE) {
    if (g_scene_ready) { memcpy(dst, g_scene, SCENE * SCENE * sizeof(float)); ok = true; }
    xSemaphoreGive(g_scene_mtx);
  }
  return ok;
}

// ---- detections ------------------------------------------------------------
void shared_set_dets(const Detection* d, int n) {
  if (n > MAX_DETS) n = MAX_DETS;
  if (xSemaphoreTake(g_dets_mtx, portMAX_DELAY) == pdTRUE) {
    for (int i = 0; i < n; i++) g_dets[i] = d[i];
    g_ndets = n;
    xSemaphoreGive(g_dets_mtx);
  }
}

int shared_get_dets(Detection* dst, int maxn) {
  int n = 0;
  if (xSemaphoreTake(g_dets_mtx, portMAX_DELAY) == pdTRUE) {
    n = g_ndets < maxn ? g_ndets : maxn;
    for (int i = 0; i < n; i++) dst[i] = g_dets[i];
    xSemaphoreGive(g_dets_mtx);
  }
  return n;
}

// ---- jpeg ------------------------------------------------------------------
void shared_set_jpeg(const uint8_t* buf, size_t len) {
  if (len > JPEG_CAP) len = JPEG_CAP;
  if (xSemaphoreTake(g_jpeg_mtx, portMAX_DELAY) == pdTRUE) {
    memcpy(g_jpeg, buf, len);
    g_jpeg_len = len;
    xSemaphoreGive(g_jpeg_mtx);
  }
}

size_t shared_get_jpeg(uint8_t* dst, size_t maxlen) {
  size_t n = 0;
  if (xSemaphoreTake(g_jpeg_mtx, portMAX_DELAY) == pdTRUE) {
    n = g_jpeg_len < maxlen ? g_jpeg_len : maxlen;
    if (n) memcpy(dst, g_jpeg, n);
    xSemaphoreGive(g_jpeg_mtx);
  }
  return n;
}

// ---- inference task (core 1) ----------------------------------------------
static int argmax(const float* v, int n) {
  int a = 0; for (int i = 1; i < n; i++) if (v[i] > v[a]) a = i; return a;
}

void inference_task(void* arg) {
  (void)arg;
  float* scene = palloc(SCENE * SCENE);
  float* crop  = palloc(CROP * CROP);
  if (!scene || !crop) { Serial.println("[infer] alloc failed"); vTaskDelete(nullptr); return; }

  Region regs[N_TILES];
  Window wins[MAX_WINDOWS];
  float  sal[N_TILES];
  Detection dets[MAX_DETS];

  for (;;) {
    if (!shared_get_scene(scene)) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }

    nn_saliency(scene, sal);
    int nr = nn_regions(sal, regs, N_TILES);
    int nw = nn_windows(sal, regs, nr, wins, MAX_WINDOWS);

    int nd = 0;
    for (int w = 0; w < nw && nd < MAX_DETS; w++) {
      vTaskDelay(pdMS_TO_TICKS(1));     // feed IDLE1 before each heavy BNN pass (WDT)
      int W0 = wins[w].x1 - wins[w].x0;
      int H0 = wins[w].y1 - wins[w].y0;
      if (W0 < 4 || H0 < 4) continue;
      nn_crop_resize(scene, wins[w].x0, wins[w].y0, W0, H0, crop, CROP);

      float probs[N_CLASS], box[4];
      nn_bnn(crop, probs, box);
      int cls = argmax(probs, N_CLASS);
      float p_face = probs[FACE_CLASS];

      // Binary head: P(bg)=1-P(face). One threshold on the face probability is
      // the whole decision -- anything below is (uncertain or) background and
      // gets no box at all.
      if (p_face < FACE_THR) continue;

      float cx = box[0], cy = box[1], bw = box[2], bh = box[3];
      dets[nd].x0 = (cx - bw / 2.0f) * W0 + wins[w].x0;
      dets[nd].y0 = (cy - bh / 2.0f) * H0 + wins[w].y0;
      dets[nd].x1 = (cx + bw / 2.0f) * W0 + wins[w].x0;
      dets[nd].y1 = (cy + bh / 2.0f) * H0 + wins[w].y0;
      dets[nd].cls = cls;
      dets[nd].conf = p_face;
      nd++;
    }
    shared_set_dets(dets, nd);
    vTaskDelay(pdMS_TO_TICKS(1));       // yield to the scheduler
  }
}
