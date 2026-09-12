// ============================================================================
//  model.cpp  -  LittleFS load + PSRAM parse of the two trained blobs.
// ============================================================================
#include "model.h"
#include "config.h"
#include <LittleFS.h>

SaliencyModel g_sal;
BnnModel      g_bnn;

// ---- helpers ---------------------------------------------------------------
static uint8_t* read_file_psram(const char* path, size_t expect, size_t* got) {
  File f = LittleFS.open(path, "r");
  if (!f) { Serial.printf("[model] open failed: %s\n", path); return nullptr; }
  size_t sz = f.size();
  if (expect && sz != expect) {
    Serial.printf("[model] %s size %u != expected %u\n", path,
                  (unsigned)sz, (unsigned)expect);
    f.close();
    return nullptr;
  }
  // 4-byte aligned PSRAM buffer so float* casts at 4-aligned offsets are safe.
  uint8_t* buf = (uint8_t*)heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
  if (!buf) buf = (uint8_t*)malloc(sz);          // fallback to internal RAM
  if (!buf) { Serial.printf("[model] alloc %u failed\n", (unsigned)sz); f.close(); return nullptr; }
  size_t rd = f.read(buf, sz);
  f.close();
  if (rd != sz) { Serial.println("[model] short read"); free(buf); return nullptr; }
  if (got) *got = sz;
  return buf;
}

static inline float softplus(float x) { return log1pf(expf(x)); }

// ---- public API ------------------------------------------------------------
bool storage_begin() {
  if (LittleFS.begin(false)) return true;       // mount existing FS
  Serial.println("[model] LittleFS mount failed, formatting...");
  return LittleFS.begin(true);                  // format then mount
}

bool model_files_present() {
  return LittleFS.exists(SALIENCY_PATH) && LittleFS.exists(BNN_PATH);
}

static bool parse_saliency() {
  size_t sz = 0;
  uint8_t* b = read_file_psram(SALIENCY_PATH, SALIENCY_BYTES, &sz);
  if (!b) return false;
  g_sal.raw = b;
  const float* p = (const float*)b;             // whole blob is float32
  g_sal.c1w = p;              p += 8 * 1 * 3 * 3;   // 72
  g_sal.c1b = p;              p += 8;               // 8
  g_sal.c2w = p;              p += 4 * 8 * 3 * 3;   // 288
  g_sal.c2b = p;              p += 4;               // 4
  g_sal.c   = p;              p += 3 * 5;           // 15
  g_sal.logsig = p;           p += 3 * 5;           // 15
  g_sal.P   = p;              p += 125 * 4;         // 500
  for (int i = 0; i < 15; i++) g_sal.sigma[i] = softplus(g_sal.logsig[i]);
  Serial.println("[model] saliency parsed");
  return true;
}

static bool parse_bnn() {
  size_t sz = 0;
  uint8_t* b = read_file_psram(BNN_PATH, BNN_BYTES, &sz);
  if (!b) return false;
  g_bnn.raw = b;
  size_t o = 0;                                  // byte offset walker
  auto R = [&](size_t n) -> const float* { const float* r = (const float*)(b + o); o += n * 4; return r; };  // NOTE: darf nicht 'F' heissen (Arduino F()-Makro)
  g_bnn.b1w  = R(360);                // 360 f  -> off 1440
  g_bnn.b1b  = R(40);                 // off 1600
  g_bnn.b2w  = R(80 * 40 * 3 * 3);    // 28800 f -> off 116800
  g_bnn.b2b  = R(80);                 // off 117120
  g_bnn.fc1b = R(192);                // off 117888
  g_bnn.fc1s = R(192);                // off 118656
  g_bnn.fc1w8 = (const int8_t*)(b + o); o += 192 * 3920;   // off 871296
  g_bnn.fc2w = R(2 * 192);            // off 872832
  g_bnn.fc2b = R(2);                  // off 872840
  g_bnn.fc3w = R(4 * 192);            // off 875912
  g_bnn.fc3b = R(4);                  // off 875928
  if (o != BNN_BYTES) {
    Serial.printf("[model] BNN parse offset %u != %u\n", (unsigned)o, BNN_BYTES);
    return false;
  }
  Serial.println("[model] bnn parsed");
  return true;
}

bool model_load_all() {
  if (!model_files_present()) {
    Serial.println("[model] blob(s) missing -- upload via /upload first");
    return false;
  }
  return parse_saliency() && parse_bnn();
}

int model_store_file(const char* path, const uint8_t* data, size_t len) {
  File f = LittleFS.open(path, "w");
  if (!f) return -1;
  size_t w = f.write(data, len);
  f.close();
  return (w == len) ? (int)w : -1;
}

void model_free_all() {
  if (g_sal.raw) { free(g_sal.raw); g_sal.raw = nullptr; }
  if (g_bnn.raw) { free(g_bnn.raw); g_bnn.raw = nullptr; }
}
