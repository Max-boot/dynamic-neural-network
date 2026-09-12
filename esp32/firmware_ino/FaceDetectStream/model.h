// ============================================================================
//  model.h  -  Load the two trained blobs from LittleFS into PSRAM and expose
//              typed pointers into them. Byte layout is fixed by the exporter
//              (see esp32/blobio/verify_blobs_final.py):
//
//   face_saliency.bin: c1w(8,1,3,3) c1b(8) c2w(4,8,3,3) c2b(4)
//                      c(3,5) log_sigma(3,5) P(125,4)              [3608 B]
//   face_bnn.bin:      b1w(40,1,3,3) b1b(40) b2w(80,40,3,3) b2b(80)
//                      fc1b(192) fc1s(192) fc1w8(192,3920) int8
//                      fc2w(2,192) fc2b(2) fc3w(4,192) fc3b(4)     [875928 B]
// ============================================================================
#pragma once
#include <Arduino.h>
#include <stdint.h>

struct SaliencyModel {
  uint8_t* raw = nullptr;      // whole-file PSRAM buffer (float-aligned)
  const float* c1w;            // (8,1,3,3)
  const float* c1b;            // (8)
  const float* c2w;            // (4,8,3,3)
  const float* c2b;            // (4)
  const float* c;              // (3,5) gaussian centers
  const float* logsig;         // (3,5) raw log-sigma
  const float* P;              // (125,4) consequents
  float sigma[15];             // softplus(logsig), precomputed
};

struct BnnModel {
  uint8_t* raw = nullptr;      // whole-file PSRAM buffer
  const float*  b1w;           // (40,1,3,3)
  const float*  b1b;           // (40)
  const float*  b2w;           // (80,40,3,3)
  const float*  b2b;           // (80)
  const float*  fc1b;          // (192)
  const float*  fc1s;          // (192) per-row dequant scale
  const int8_t* fc1w8;         // (192,3920)
  const float*  fc2w;          // (2,192)
  const float*  fc2b;          // (2)
  const float*  fc3w;          // (4,192)
  const float*  fc3b;          // (4)
};

// Global model instances (defined in model.cpp).
extern SaliencyModel g_sal;
extern BnnModel      g_bnn;

// Mount LittleFS. Returns false on failure.
bool storage_begin();

// True if both blob files exist on LittleFS with the expected sizes.
bool model_files_present();

// Parse both blobs from LittleFS into PSRAM. Returns false on any error
// (missing file, wrong size, or allocation failure).
bool model_load_all();

// Persist an uploaded blob to LittleFS (used by the /upload endpoint before a
// reboot re-parses it). Returns bytes written, or -1 on error.
int  model_store_file(const char* path, const uint8_t* data, size_t len);

// Free PSRAM buffers (rarely needed; reboot is the normal path).
void model_free_all();
