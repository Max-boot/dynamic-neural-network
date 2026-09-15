// ============================================================================
//  pipeline.cpp  -  Shared state hub + inference task (runs on core 1).
// ============================================================================
#include "pipeline.h"
#include "config.h"
#include "model.h"
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

// ---- shared buffers --------------------------------------------------------
static float*   g_scene = nullptr;        // latest RGB scene [N_CH*SCENE*SCENE]
static bool     g_scene_ready = false;
static SemaphoreHandle_t g_scene_mtx = nullptr;

static Detection g_dets[MAX_DETS];
static int       g_ndets = 0;
static SemaphoreHandle_t g_dets_mtx = nullptr;

#define JPEG_CAP (64 * 1024)              // roomy for a hardware-QVGA frame
static uint8_t* g_jpeg = nullptr;
static size_t   g_jpeg_len = 0;
static SemaphoreHandle_t g_jpeg_mtx = nullptr;

static float* palloc(size_t n) {
  float* p = (float*)heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!p) p = (float*)malloc(n * sizeof(float));
  return p;
}

bool shared_begin() {
  g_scene = palloc(N_CH * SCENE * SCENE);
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
    memcpy(g_scene, scene, N_CH * SCENE * SCENE * sizeof(float));
    g_scene_ready = true;
    xSemaphoreGive(g_scene_mtx);
  }
}

bool shared_get_scene(float* dst) {
  bool ok = false;
  if (xSemaphoreTake(g_scene_mtx, portMAX_DELAY) == pdTRUE) {
    if (g_scene_ready) { memcpy(dst, g_scene, N_CH * SCENE * SCENE * sizeof(float)); ok = true; }
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
  float* scene = palloc(N_CH * SCENE * SCENE);
  float* crop  = palloc(N_CH * CROP * CROP);
  if (!scene || !crop) { Serial.println("[infer] alloc failed"); vTaskDelete(nullptr); return; }

  Region regs[N_TILES];
  Window wins[MAX_WINDOWS];
  float  sal[N_TILES];
  Detection dets[MAX_DETS];

  unsigned long frame_cnt = 0;
  for (;;) {
    if (!nn_model_ready()) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }

    if (!shared_get_scene(scene)) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }

    unsigned long t0 = millis();

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

      if (p_face < FACE_THR) continue;

      // Student: box = full adaptive window (no fc3 refinement).
      // Tighter box comes from tighter saliency windows, not from the network.
      dets[nd].x0 = wins[w].x0;
      dets[nd].y0 = wins[w].y0;
      dets[nd].x1 = wins[w].x1;
      dets[nd].y1 = wins[w].y1;
      dets[nd].cls = cls;
      dets[nd].conf = p_face;
      nd++;
    }
    shared_set_dets(dets, nd);

    unsigned long dt = millis() - t0;
    frame_cnt++;
    if ((frame_cnt & 0x1F) == 0) {    // every 32 frames
      Serial.printf("[infer] %lu ms  wins=%d dets=%d\n", dt, nw, nd);
    }
    vTaskDelay(pdMS_TO_TICKS(1));       // yield to the scheduler
  }
}
