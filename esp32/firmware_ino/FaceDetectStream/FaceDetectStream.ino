// ============================================================================
//  FaceDetectStream.ino  -  AI-Thinker ESP32-CAM face-detection livestream.
//
//  Architecture (dual core, FreeRTOS):
//    Core 0  camera_task : owns the camera. Captures a QVGA JPEG frame
//            (PIXFORMAT_JPEG, hardware-encoded), decodes to RGB888, builds the
//            3-channel 128x128 model scene, draws detection boxes on the
//            RGB888 buffer, encodes back to JPEG, publishes for streaming.
//    Core 0  http server : streams the published JPEG as MJPEG (esp_http_server,
//            pinned to core 0 in web.cpp).
//    Core 1  inference_task : reads a scene snapshot, runs the full cascade
//            (Conv+MLP saliency -> regions -> refocus windows -> int8 BNN +
//            box head), publishes detections. Never touches the camera.
//
//  Only the camera task calls esp_camera_fb_get/return, so there is never
//  concurrent frame-buffer access. All cross-core data goes through the mutex-
//  guarded hub in pipeline.cpp.
//
//  Board  : "AI Thinker ESP32-CAM"     Partition: "Huge APP (3MB No OTA/1MB SPIFFS)"
//  PSRAM  : enabled                     Flash: 4MB (QIO 40MHz)
// ============================================================================
#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"
#include <esp_heap_caps.h>

#include "config.h"
#include "camera_pins.h"
#include "model.h"
#include "nn.h"
#include "pipeline.h"
#include "web.h"

// ---- decode scratch: camera JPEG -> RGB888 (QVGA 320x240), model input ----
static uint8_t* g_rgb888 = nullptr;         // [CAM_W*CAM_H*3]
static float*   g_scene_local = nullptr;    // [N_CH*SCENE*SCENE] scratch for capture
static bool     g_model_ok = false;

// ---- camera init -----------------------------------------------------------
static bool camera_init() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = CAM_XCLK_HZ;
  c.pixel_format    = PIXFORMAT_JPEG;        // HARDWARE JPEG encode, no fmt2jpg
  c.frame_size      = FRAMESIZE_QVGA;        // 320x240
  c.jpeg_quality    = JPEG_QUALITY;
  c.fb_count        = 2;
  c.fb_location     = CAMERA_FB_IN_PSRAM;
  c.grab_mode       = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) { Serial.printf("[cam] init failed 0x%x\n", err); return false; }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    // Kamera ist physisch kopfüber montiert. Statt die Szene software-seitig
    // zu spiegeln, flippt der Sensor direkt -> JPEG, Scene und Boxen sind
    // identisch ausgerichtet, render_frame mappt 1:1.
    s->set_vflip(s, 1);
    s->set_hmirror(s, 0);
  }
  return true;
}

// ---- scene build: RGB888 frame -> center-cropped 128x128 float, 3 planes ----
// Mirrors sim_pipeline.load_rgb_scene: center square crop, bilinear resize to
// SCENE, /255. linspace(0, S-1, SCENE) sampling. Scene layout is channel-major
// [R][G][B] planes, each [SCENE*SCENE] row-major, matching nn_saliency(C=N_CH).
// Input is the fmt2rgb888-decoded camera JPEG (<-> sensor, already vflip=1 so
// no manual mirror here).
static void build_scene(const uint8_t* g8, int W, int H, float* scene) {
  int S   = W < H ? W : H;                 // square side
  int ox  = (W - S) / 2;
  int oy  = (H - S) / 2;
  float step = (S > 1) ? (float)(S - 1) / (float)(SCENE - 1) : 0.0f;
  for (int j = 0; j < SCENE; j++) {
    float fy = j * step;
    int y0 = (int)fy; int y1 = y0 + 1; if (y1 > S - 1) y1 = S - 1;
    float wy = fy - y0;
    for (int i = 0; i < SCENE; i++) {
      float fx = i * step;
      int x0 = (int)fx; int x1 = x0 + 1; if (x1 > S - 1) x1 = S - 1;
      float wx = fx - x0;
      // 4 Ecken lesen (RGB888) und je Kanal bilinear interpolieren.
      float rgb[3][4];                            // [channel][corner]
      for (int q = 0; q < 4; q++) {
        int col = (q & 1) ? x1 : x0;
        int row = (q & 2) ? y1 : y0;
        const uint8_t* px = g8 + ((oy + row) * W + ox + col) * 3;  // 3 B/px
        rgb[0][q] = (float)px[0];
        rgb[1][q] = (float)px[1];
        rgb[2][q] = (float)px[2];
      }
      for (int c = 0; c < N_CH; c++) {
        float w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy);
        float w10 = (1 - wx) * wy,      w11 = wx * wy;
        float v = rgb[c][0] * w00 + rgb[c][1] * w01
                + rgb[c][2] * w10 + rgb[c][3] * w11;
        scene[c * SCENE * SCENE + j * SCENE + i] = v / 255.0f;
      }
    }
  }
}

// ---- overlay on the decoded RGB888 frame (QVGA 320x240) -------------------
static inline void putpx(uint8_t* rgb, int W, int H, int x, int y,
                         uint8_t r, uint8_t gg, uint8_t b) {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  uint8_t* p = rgb + (y * W + x) * 3;
  p[0] = r; p[1] = gg; p[2] = b;
}

static void draw_rect(uint8_t* rgb, int W, int H,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t gg, uint8_t b) {
  for (int t = 0; t < 2; t++) {              // 2px thick
    for (int x = x0; x <= x1; x++) { putpx(rgb, W, H, x, y0 + t, r, gg, b); putpx(rgb, W, H, x, y1 - t, r, gg, b); }
    for (int y = y0; y <= y1; y++) { putpx(rgb, W, H, x0 + t, y, r, gg, b); putpx(rgb, W, H, x1 - t, y, r, gg, b); }
  }
}

// Boxes are in scene units (0..SCENE); map to camera pixels exactly like
// build_scene: center square crop at (ox,oy) of side S, step=(S-1)/(SCENE-1).
static void render_frame(uint8_t* rgb, int W, int H, const Detection* d, int nd) {
  int S   = W < H ? W : H;
  int ox  = (W - S) / 2;
  int oy  = (H - S) / 2;
  float step = (S > 1) ? (float)(S - 1) / (float)(SCENE - 1) : 0.0f;

  for (int i = 0; i < nd; i++) {
    int x0 = ox + (int)(d[i].x0 * step + 0.5f);
    int y0 = oy + (int)(d[i].y0 * step + 0.5f);
    int x1 = ox + (int)(d[i].x1 * step + 0.5f);
    int y1 = oy + (int)(d[i].y1 * step + 0.5f);
    if (d[i].cls == FACE_CLASS) draw_rect(rgb, W, H, x0, y0, x1, y1, 0, 255, 0);
    else                        draw_rect(rgb, W, H, x0, y0, x1, y1, 255, 160, 0);
  }
}

// ---- camera task (core 0) --------------------------------------------------
// The stream must show a PERSISTENT box while a detection exists. That is only
// possible if frames without a box are published as raw camera JPEG and frames
// WITH a box are published as the drawn+re-encoded JPEG. Two subtleties:
//   * never publish the raw frame first and the drawn one later, otherwise the
//     next loop's raw publish instantly erases the box (flicker);
//   * the camera FB is held until the last use (fb_return after the branch).
//
// fmt2rgb888 in this esp32-camera build emits BGR for a JPEG source while
// build_scene and fmt2jpg both expect RGB. The raw fast path is unaffected
// (native camera JPEG), but the re-encode path would swap R<->B (red tones
// turn blue, green boxes stay green because 0,255,0 is symmetric). Fix: swap
// channels 0<->2 once after the decode so scene + stream use correct RGB.
static void bgr_to_rgb(uint8_t* buf, size_t npix) {
  for (size_t i = 0; i < npix; i++, buf += 3) {
    uint8_t t = buf[0]; buf[0] = buf[2]; buf[2] = t;
  }
}
static void camera_task(void* arg) {
  (void)arg;
  Detection dets[MAX_DETS];
  unsigned long cam_cnt = 0;
  for (;;) {
    unsigned long tc0 = millis();
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

    // Model input: JPEG -> RGB888 decode (fb still held for the raw path).
    fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, g_rgb888);
    bgr_to_rgb(g_rgb888, CAM_W * CAM_H);

    build_scene(g_rgb888, CAM_W, CAM_H, g_scene_local);
    shared_set_scene(g_scene_local);

    int nd = shared_get_dets(dets, MAX_DETS);
    unsigned long tc1 = millis();
    if (nd > 0) {
      // Face present: draw boxes on the decoded RGB888 and re-encode. The
      // overlaid JPEG is the ONLY frame published this loop -> box persists.
      render_frame(g_rgb888, CAM_W, CAM_H, dets, nd);
      esp_camera_fb_return(fb);

      uint8_t* jpg = nullptr; size_t jlen = 0;
      if (fmt2jpg(g_rgb888, CAM_W * CAM_H * 3, CAM_W, CAM_H, PIXFORMAT_RGB888, JPEG_QUALITY, &jpg, &jlen)) {
        shared_set_jpeg(jpg, jlen);
        free(jpg);
      }
      unsigned long tc2 = millis();
      cam_cnt++;
      if ((cam_cnt & 0x1F) == 0) {
        Serial.printf("[cam] decode+scene=%lu jpg=%lu total=%lu dets=%d\n",
                      tc1-tc0, tc2-tc1, tc2-tc0, nd);
      }
    } else {
      // No face: stream the camera's OWN hardware JPEG, zero software encode.
      shared_set_jpeg(fb->buf, fb->len);
      esp_camera_fb_return(fb);

      cam_cnt++;
      if ((cam_cnt & 0x1F) == 0) {
        Serial.printf("[cam] cap+dec=%lu total=%lu dets=0 (fast path)\n", tc1-tc0, millis()-tc0);
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ---- WiFi ------------------------------------------------------------------
static String wifi_up() {
  if (WIFI_AP_MODE) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, strlen(AP_PASS) >= 8 ? AP_PASS : nullptr);
    IPAddress ip = WiFi.softAPIP();
    Serial.printf("[wifi] AP \"%s\"  http://%s/\n", AP_SSID, ip.toString().c_str());
    return ip.toString();
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);
  Serial.printf("[wifi] joining \"%s\"", STA_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) { delay(400); Serial.print('.'); }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) { Serial.println("[wifi] failed"); return String("0.0.0.0"); }
  IPAddress ip = WiFi.localIP();
  Serial.printf("[wifi] STA  http://%s/\n", ip.toString().c_str());
  return ip.toString();
}

// ---- setup / loop ----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(300);
  Serial.println("\n[boot] FaceDetectStream");

  if (!psramFound()) { Serial.println("[boot] FATAL: no PSRAM"); }

  // Decode scratch (camera JPEG -> RGB888, QVGA) + capture scratch (PSRAM).
  g_rgb888 = (uint8_t*)heap_caps_malloc(CAM_W * CAM_H * 3, MALLOC_CAP_SPIRAM);
  g_scene_local = (float*)heap_caps_malloc(N_CH * SCENE * SCENE * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!g_rgb888 || !g_scene_local) { Serial.println("[boot] FATAL: scratch alloc"); }

  if (!camera_init())  Serial.println("[boot] camera init FAILED");
  if (!storage_begin()) Serial.println("[boot] storage init FAILED");
  if (!shared_begin())  Serial.println("[boot] shared init FAILED");
  if (!nn_begin())      Serial.println("[boot] nn scratch FAILED");

  g_model_ok = model_load_all();
  if (!g_model_ok)
    Serial.println("[boot] model not loaded -- upload blobs via web UI or upload_model.py, then reboot");

  String ip = wifi_up();
  web_set_status(g_model_ok, ip.c_str());
  web_begin();

  // Inference on core 1; camera + stream on core 0. The inference task
  // dereferences the model pointers on the very first frame, so never start
  // it unless the blobs actually loaded (see pipeline.cpp guard as the
  // second line of defence).
  if (g_model_ok)
    xTaskCreatePinnedToCore(inference_task, "infer", 32768, nullptr, 1, nullptr, 1);
  else
    Serial.println("[boot] inference disabled -- model not loaded, upload blobs then reboot");
  xTaskCreatePinnedToCore(camera_task,    "cam",   16384, nullptr, 2, nullptr, 0);

  Serial.println("[boot] running");
}

void loop() {
  // Everything runs in the pinned tasks. Idle-report every few seconds.
  static uint32_t last = 0;
  if (millis() - last > 5000) {
    last = millis();
    Serial.printf("[stat] heap=%u psram=%u model=%d\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                  g_model_ok);
  }
  delay(1000);
}
