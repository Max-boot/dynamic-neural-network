// ============================================================================
//  FaceDetectStream.ino  -  AI-Thinker ESP32-CAM face-detection livestream.
//
//  Architecture (dual core, FreeRTOS):
//    Core 0  camera_task : owns the camera. Captures a color QVGA frame (RGB565,
//            but the OV2640 has no auto-color so it is nominal), builds the
//            3-channel 128x128 model scene (R,G,B planes), publishes it, reads
//            the latest detections, upscales + draws boxes, JPEG-encodes,
//            publishes JPEG.
//    Core 0  http server : streams the published JPEG as MJPEG (esp_http_server,
//            pinned to core 0 in web.cpp).
//    Core 1  inference_task : reads a scene snapshot, runs the full cascade
//            (Conv+ANFIS saliency -> regions -> refocus windows -> int8 BNN +
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

// ---- display buffer (RGB888, upscaled scene the boxes are drawn on) --------
#define DISP  (SCENE * STREAM_SCALE)          // 256
static uint8_t* g_rgb = nullptr;              // [DISP*DISP*3]
static float*   g_scene_local = nullptr;      // [N_CH*SCENE*SCENE] scratch for capture
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
  c.pixel_format = PIXFORMAT_RGB565;   // 2 bytes/pixel -> color model input (R,G,B)
  c.frame_size   = FRAMESIZE_QVGA;     // 320x240
  c.fb_count     = 2;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) { Serial.printf("[cam] init failed 0x%x\n", err); return false; }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
  }
  return true;
}

// ---- scene build: RGB565 frame -> center-cropped 128x128 float, 3 planes ----
// Mirrors sim_pipeline.load_rgb_scene: center square crop, bilinear resize to
// SCENE, /255. linspace(0, S-1, SCENE) sampling. Scene layout is channel-major
// [R][G][B] planes, each [SCENE*SCENE] row-major, matching nn_saliency(C=N_CH).
// RGB565 -> 0..255 exakte lineare Skalierung (255/31 bzw. 255/63 pro Bit):
static void build_scene(const uint8_t* g8, int W, int H, float* scene) {
  const uint16_t* g = (const uint16_t*)g8;    // little-endian host: byte0 = LSB
  int S   = W < H ? W : H;                 // square side
  int ox  = (W - S) / 2;
  int oy  = (H - S) / 2;
  float step = (S > 1) ? (float)(S - 1) / (float)(SCENE - 1) : 0.0f;
  for (int j = 0; j < SCENE; j++) {
    float fy = j * step;
    int y0 = (int)fy; int y1 = y0 + 1; if (y1 > S - 1) y1 = S - 1;
    float wy = fy - y0;
    const uint16_t* r0 = g + (oy + y0) * W + ox;
    const uint16_t* r1 = g + (oy + y1) * W + ox;
    for (int i = 0; i < SCENE; i++) {
      float fx = i * step;
      int x0 = (int)fx; int x1 = x0 + 1; if (x1 > S - 1) x1 = S - 1;
      float wx = fx - x0;
      // 4 Ecken dekodieren und je Kanal bilinear interpolieren.
      float rgb[3][4];                            // [channel][corner]
      for (int q = 0; q < 4; q++) {
        int col = (q & 1) ? x1 : x0;
        int row = (q & 2) ? y1 : y0;
        const uint16_t* px = (row == y0 ? r0 : r1) + col;
        uint16_t p565 = *px;
        float r8 = float((p565 >> 11) & 0x1F) * (255.0f / 31.0f);
        float g8 = float((p565 >> 5)  & 0x3F) * (255.0f / 63.0f);
        float b8 = float((p565)       & 0x1F) * (255.0f / 31.0f);
        rgb[0][q] = r8; rgb[1][q] = g8; rgb[2][q] = b8;
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

// ---- overlay ---------------------------------------------------------------
static inline void putpx(int x, int y, uint8_t r, uint8_t gg, uint8_t b) {
  if (x < 0 || x >= DISP || y < 0 || y >= DISP) return;
  uint8_t* p = g_rgb + (y * DISP + x) * 3;
  p[0] = r; p[1] = gg; p[2] = b;
}

static void draw_rect(int x0, int y0, int x1, int y1, uint8_t r, uint8_t gg, uint8_t b) {
  for (int t = 0; t < 2; t++) {              // 2px thick
    for (int x = x0; x <= x1; x++) { putpx(x, y0 + t, r, gg, b); putpx(x, y1 - t, r, gg, b); }
    for (int y = y0; y <= y1; y++) { putpx(x0 + t, y, r, gg, b); putpx(x1 - t, y, r, gg, b); }
  }
}

static void render_frame(const float* scene, const Detection* d, int nd) {
  // Upscale RGB scene ([N_CH*SCENE*SCENE], planes R,G,B) -> RGB888 display (nearest).
  for (int y = 0; y < DISP; y++) {
    int sy = y / STREAM_SCALE;
    for (int x = 0; x < DISP; x++) {
      int sx = x / STREAM_SCALE;
      uint8_t* p = g_rgb + (y * DISP + x) * 3;
      for (int c = 0; c < 3; c++) {
        float v = scene[c * SCENE * SCENE + sy * SCENE + sx];
        p[c] = (uint8_t)(v * 255.0f + 0.5f);
      }
    }
  }
  // Boxes: green for FACE_CLASS, orange otherwise. Coords are scene units.
  for (int i = 0; i < nd; i++) {
    int x0 = (int)(d[i].x0 * STREAM_SCALE + 0.5f);
    int y0 = (int)(d[i].y0 * STREAM_SCALE + 0.5f);
    int x1 = (int)(d[i].x1 * STREAM_SCALE + 0.5f);
    int y1 = (int)(d[i].y1 * STREAM_SCALE + 0.5f);
    if (d[i].cls == FACE_CLASS) draw_rect(x0, y0, x1, y1, 0, 255, 0);
    else                        draw_rect(x0, y0, x1, y1, 255, 160, 0);
  }
}

// ---- camera task (core 0) --------------------------------------------------
static void camera_task(void* arg) {
  (void)arg;
  Detection dets[MAX_DETS];
  for (;;) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

    build_scene(fb->buf, fb->width, fb->height, g_scene_local);
    esp_camera_fb_return(fb);                 // release ASAP; scene is copied out

    shared_set_scene(g_scene_local);          // hand to inference (core 1)

    int nd = shared_get_dets(dets, MAX_DETS); // latest results (may lag a frame)
    render_frame(g_scene_local, dets, nd);

    uint8_t* jpg = nullptr; size_t jlen = 0;
    if (fmt2jpg(g_rgb, DISP * DISP * 3, DISP, DISP, PIXFORMAT_RGB888, JPEG_QUALITY, &jpg, &jlen)) {
      shared_set_jpeg(jpg, jlen);
      free(jpg);
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

  // Display + capture scratch (PSRAM).
  g_rgb = (uint8_t*)heap_caps_malloc(DISP * DISP * 3, MALLOC_CAP_SPIRAM);
  g_scene_local = (float*)heap_caps_malloc(N_CH * SCENE * SCENE * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!g_rgb || !g_scene_local) { Serial.println("[boot] FATAL: scratch alloc"); }

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
