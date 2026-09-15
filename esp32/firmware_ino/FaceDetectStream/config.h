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
#define WIFI_AP_MODE      0
#define AP_SSID           "ESP32-FaceCam"
#define AP_PASS           "facecam123"        // >= 8 chars, or "" for open AP
#define STA_SSID          "HUA"
#define STA_PASS          "9927210000"
#define HTTP_PORT         80

// ---------------------------------------------------------------------------
// Model files (uploaded to LittleFS via tools/upload_model.py or the web UI)
// ---------------------------------------------------------------------------
#define SALIENCY_PATH     "/face_saliency.bin"
#define BNN_PATH          "/face_bnn.bin"
#define SALIENCY_BYTES    2964                 // sanity-check sizes on load
                                               // (MLP head: conv 8/4ch + fc 12-16-1)
#define BNN_BYTES         227112                // student blob (no box head)

// ---------------------------------------------------------------------------
// Pipeline geometry (fixed by the trained model -- do NOT change)
// ---------------------------------------------------------------------------
#define N_CH              3      // model input channels (RGB)
#define SCENE             128    // model input side
#define TILE              16
#define GRID              8      // SCENE / TILE
#define N_TILES           64     // GRID * GRID
#define CROP              28     // BNN input side
#define N_CLASS           2      // face model: 2 logits
#define BNN_C1            24     // conv1 output channels (student: 24)
#define BNN_C2            40     // conv2 output channels (student: 40)
#define BNN_FLAT          1960   // BNN_C2 * 7 * 7
#define BNN_HIDDEN        96     // fc1 output (student: 96)

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
//   REGION_MODE_ADAPTIVE : two-gate proposer (DEFAULT; matches sim --mode adaptive).
//        Gate 1 (nn_regions): adaptive HYSTERESIS on the 8x8 saliency map decides
//        WHERE to look. The number of windows is data-driven -- 0 on an empty
//        scene, more only when the evidence is there; never a fixed count.
//        A single large face stays ONE window; two faces separated by a saliency
//        valley split into two (peak split). Windows are rectangular and sized to
//        the region (very adaptive), not clamped to a fixed square.
//        Gate 2 (pipeline.cpp): P(face) >= FACE_THR decides WHAT to draw -> as few
//        boxes as possible without missing a face.
//   REGION_MODE_REFOCUS  : legacy square crop, side clamped to [WIN_MIN,WIN_MAX].
//   REGION_MODE_BBOX     : legacy raw region bounding box (+BBOX_MARGIN).
// ---------------------------------------------------------------------------
#define REGION_MODE_ADAPTIVE 2
#define REGION_MODE_REFOCUS  0
#define REGION_MODE_BBOX     1
#define REGION_MODE          REGION_MODE_ADAPTIVE

// --- Gate 1: adaptive hysteresis proposer (REGION_MODE_ADAPTIVE) ------------
// Per-frame thresholds from the 64-tile saliency mean (mu) and sample std (sd):
//   T_high = clamp(mu + SEED_K*sd, SEED_FLOOR, SEED_CEIL)  // confirms a region
//   T_low  = clamp(mu + GROW_K*sd, GROW_FLOOR, T_high)     // grows its extent
// SEED_FLOOR keeps flat / empty scenes from seeding noise; SEED_CEIL guarantees a
// very confident (e.g. full-frame) face still seeds even when sd is tiny.
#define SEED_K            2.0f
#define GROW_K            0.5f
#define SEED_FLOOR        0.55f
#define SEED_CEIL         0.80f
#define GROW_FLOOR        0.40f
// Peak split (valley/prominence based, NOT spatial): within one region a second
// peak is kept only if it stands at least PROMINENCE (saliency) above the saddle
// that connects it to a stronger peak. A flat plateau (big face) has saddle==peak
// -> prominence 0 -> stays ONE peak; two hills across a real saliency valley have
// a deep saddle -> both survive -> the region splits. Larger => harder to split.
#define PROMINENCE        0.15f
// Window sizing: symmetric margin as a percent of the region side (integer math,
// bit-identical to the sim), then a minimum side so the BNN always gets enough
// pixels. The ceiling is the scene edge -> size is fully adaptive upward.
#define MARGIN_PCT        5
#define WIN_FLOOR         24

// --- legacy (REGION_MODE_REFOCUS / _BBOX only) ------------------------------
#define WIN_MIN           24
#define WIN_MAX           48
#define BBOX_MARGIN       2

// Safety ceiling on windows / BNN passes per frame (each pass is heavy on a plain
// ESP32). With the adaptive proposer this is rarely reached -- it is a compute
// guard, NOT a target. Regions are sorted by saliency, so the strongest win.
#define MAX_WINDOWS       4
#define MAX_DETS          8      // detections retained for overlay

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------
// JPEG_QUALITY goes to the OV2640 HARDWARE encoder (camera produces the JPEG
// directly, no software fmt2jpg). Lower = better quality / bigger frame.
#define JPEG_QUALITY      12
#define CAM_XCLK_HZ       20000000
#define CAM_W             320              // QVGA, must match FRAMESIZE_QVGA
#define CAM_H             240
