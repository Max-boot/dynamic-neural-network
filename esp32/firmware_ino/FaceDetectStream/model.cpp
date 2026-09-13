// ============================================================================
//  model.cpp  -  Embedded blob access: the two trained blobs are compiled into
//  the firmware as PROGMEM arrays (see gen_blob_headers.py / *_bin_data.h), so
//  no LittleFS upload is needed. The pointers below view into those arrays.
// ============================================================================
#include "model.h"
#include "config.h"
#include "face_saliency_bin_data.h"
#include "face_bnn_bin_data.h"
#include <LittleFS.h>

SaliencyModel g_sal;
BnnModel      g_bnn;

// ---- public API ------------------------------------------------------------
bool storage_begin() {
  // LittleFS still mounted for the (legacy) web upload endpoint even though
  // the models are now embedded; failures here are non-fatal.
  if (LittleFS.begin(false)) return true;
  return LittleFS.begin(true);
}

bool model_files_present() { return true; }  // blobs are baked into the image

static bool parse_saliency() {
  // PROGMEM array is __attribute__((aligned(4))), so the float* view is safe.
  const float* p = (const float*)face_saliency_data;
  g_sal.raw = (const uint8_t*)face_saliency_data;
  g_sal.c1w = p;              p += 8 * 3 * 3 * 3;   // 216
  g_sal.c1b = p;              p += 8;               // 8
  g_sal.c2w = p;              p += 4 * 8 * 3 * 3;   // 288
  g_sal.c2b = p;              p += 4;               // 4
  g_sal.fc1w = p;             p += 16 * 12;         // 192
  g_sal.fc1b = p;             p += 16;              // 16
  g_sal.fc2w = p;             p += 1 * 16;          // 16
  g_sal.fc2b = p;             p += 1;               // 1
  if (FACE_SALIENCY_BYTES != SALIENCY_BYTES) {
    Serial.printf("[model] saliency embedded %u != expect %u\n",
                  (unsigned)FACE_SALIENCY_BYTES, (unsigned)SALIENCY_BYTES);
    return false;
  }
  Serial.println("[model] saliency parsed (embedded)");
  return true;
}

static bool parse_bnn() {
  const uint8_t* b = face_bnn_data;          // PROGMEM, 4-aligned
  g_bnn.raw = b;
  size_t o = 0;                              // byte offset walker
  auto R = [&](size_t n) -> const float* { const float* r = (const float*)(b + o); o += n * 4; return r; };  // NOTE: darf nicht 'F' heissen (Arduino F()-Makro)
  g_bnn.b1w  = R(40 * 3 * 3 * 3);     // 1080 f -> off 4320
  g_bnn.b1b  = R(40);                 // off 4480
  g_bnn.b2w  = R(80 * 40 * 3 * 3);    // 28800 f -> off 119680
  g_bnn.b2b  = R(80);                 // off 120000
  g_bnn.fc1b = R(192);                // off 120768
  g_bnn.fc1s = R(192);                // off 121536
  g_bnn.fc1w8 = (const int8_t*)(b + o); o += 192 * 3920;   // off 874176
  g_bnn.fc2w = R(2 * 192);            // off 875712
  g_bnn.fc2b = R(2);                  // off 875720
  g_bnn.fc3w = R(4 * 192);            // off 878792
  g_bnn.fc3b = R(4);                  // off 878808
  if (o != BNN_BYTES) {
    Serial.printf("[model] BNN parse offset %u != %u\n", (unsigned)o, BNN_BYTES);
    return false;
  }
  Serial.println("[model] bnn parsed (embedded)");
  return true;
}

bool model_load_all() {
  return parse_saliency() && parse_bnn();
}

bool nn_model_ready() {
  return g_sal.raw != nullptr && g_bnn.raw != nullptr;
}

int model_store_file(const char* path, const uint8_t* data, size_t len) {
  (void)path; (void)data; (void)len;
  Serial.println("[model] store disabled -- blobs are embedded");
  return -1;
}

void model_free_all() {
  // Embedded blobs live in PROGMEM/flash: nothing to free.
  g_sal.raw = nullptr;
  g_bnn.raw = nullptr;
}
