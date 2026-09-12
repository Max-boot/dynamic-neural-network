// ============================================================================
//  config.h  -  FaceDetectStream (AI-Thinker ESP32-CAM)
// ----------------------------------------------------------------------------
//  All user-tunable parameters live here. The pipeline math mirrors
//  esp32/tools/sim_pipeline.py 1:1 -- calibrate the thresholds on the host
//  simulator first (run it with --mode refocus to match this firmware), then
//  copy the values you validated into this file.
// ============================================================================
#pragma once

// ---------------------------------------------------------------------------
// WiFi
//   MODE_AP   = 1 : ESP32 opens its own access point (connect your phone/PC to
//                   it, then open http://192.168.4.1/).  No router needed.
//   MODE_AP   = 0 : ESP32 joins your existing WiFi (fill STA_SSID / STA_PASS);
//                   the chosen IP is printed on the serial monitor.
// ---------------------------------------------------------------------------
#define WIFI_AP_MODE      1
#define AP_SSID           "ESP32-FaceCam"
#define AP_PASS           "facecam123"        // >= 8 chars, or "" for open AP
#define STA_SSID          "your-wifi"
#define STA_PASS          "your-password"
#define HTTP_PORT         80

// ---------------------------------------------------------------------------
// Model files (uploaded to LittleFS via tools/upload_model.py or the web UI)
// ---------------------------------------------------------------------------
#define SALIENCY_PATH     "/face_saliency.bin"
#define BNN_PATH          "/face_bnn.bin"
#define SALIENCY_BYTES    3608                 // sanity-check sizes on load
#define BNN_BYTES         875928

// ---------------------------------------------------------------------------
// Pipeline geometry (fixed by the trained model -- do NOT change)
// ---------------------------------------------------------------------------
#define SCENE             128    // model input side
#define TILE              16
#define GRID              8      // SCENE / TILE
#define N_TILES           64     // GRID * GRID
#define CROP              28     // BNN input side
#define N_CLASS           2      // face model: 2 logits
#define BNN_FLAT          3920   // 80 * 7 * 7

// ---------------------------------------------------------------------------
// Detection thresholds  (calibrate with sim_pipeline.py, then paste here)
// ---------------------------------------------------------------------------
#define REGION_THR        0.5f   // saliency tile counts as active if >= this
#define FACE_THR          0.6f   // draw a box iff P(face) >= this. Binary head:
                                 // P(bg)=1-P(face), so one threshold on the face
                                 // probability is the whole decision -- background
                                 // is never drawn, only rejected.

// Class index mapping -- ASSUMPTION until calibrated on the simulator!
// Run sim_pipeline.py on a real face photo and a non-face photo; whichever
// index is high for faces and low otherwise is FACE_CLASS.
#define FACE_CLASS        0
#define BG_CLASS          1

// ---------------------------------------------------------------------------
// Region -> window strategy
//   REGION_MODE_REFOCUS : one square crop per connected region, centered on the
//                         saliency centroid, side = region extent clamped to
//                         [WIN_MIN, WIN_MAX]. Best for single-subject faces and
//                         bounds compute. Matches sim_pipeline.py --mode refocus.
//   REGION_MODE_BBOX    : raw region bounding box (+margin). --mode bbox.
// ---------------------------------------------------------------------------
#define REGION_MODE_REFOCUS  0
#define REGION_MODE_BBOX     1
#define REGION_MODE          REGION_MODE_REFOCUS
#define WIN_MIN           24
#define WIN_MAX           48
#define BBOX_MARGIN       2

// Cap windows evaluated per frame (each BNN pass is heavy on a plain ESP32).
// Regions are sorted largest-first, so the most salient blobs win.
#define MAX_WINDOWS       4
#define MAX_DETS          8      // detections retained for overlay

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------
#define JPEG_QUALITY      12     // 0..63, lower = better quality / bigger
#define STREAM_SCALE      2      // output JPEG is SCENE*STREAM_SCALE per side
#define CAM_XCLK_HZ       20000000
