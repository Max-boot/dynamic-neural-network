// ============================================================================
//  pipeline.h  -  Shared state between the two cores + the inference task.
//
//  Core 0 (camera task):  writes the latest 128x128 gray scene and the latest
//                         overlaid JPEG.
//  Core 1 (inference task): reads a scene snapshot, runs the full cascade, and
//                         writes the detection list.
//  Core 0 (web/stream):   reads the latest JPEG.
//  All cross-core state is guarded by mutexes here.
// ============================================================================
#pragma once
#include <Arduino.h>
#include "nn.h"

// Allocate shared buffers + mutexes. Call once at boot.
bool shared_begin();

// --- scene (camera -> inference) -------------------------------------------
void shared_set_scene(const float* scene /*[SCENE*SCENE]*/);
// Copies the latest scene into dst; returns false if no frame yet.
bool shared_get_scene(float* dst /*[SCENE*SCENE]*/);

// --- detections (inference -> overlay) -------------------------------------
void shared_set_dets(const Detection* d, int n);
int  shared_get_dets(Detection* dst, int maxn);   // returns count copied

// --- latest JPEG (camera -> stream) ----------------------------------------
void   shared_set_jpeg(const uint8_t* buf, size_t len);
size_t shared_get_jpeg(uint8_t* dst, size_t maxlen);  // returns bytes copied

// Inference loop (pin to core 1). Never returns.
void inference_task(void* arg);
