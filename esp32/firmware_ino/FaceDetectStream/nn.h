// ============================================================================
//  nn.h  -  Forward-pass primitives for the face cascade. Every routine mirrors
//           esp32/tools/sim_pipeline.py so the firmware can be audited line by
//           line against that (verified) NumPy reference.
// ============================================================================
#pragma once
#include <Arduino.h>
#include "config.h"

struct Region  { int x0, y0, x1, y1, tiles; };
struct Window  { int x0, y0, x1, y1; };
struct Detection { float x0, y0, x1, y1; int cls; float conf; };

// Allocate all PSRAM scratch buffers. Call once at boot AFTER psramInit().
bool nn_begin();

// Stage 1+2: scene[SCENE*SCENE] (float 0..1, row-major) -> sal[N_TILES] (8x8,
// row-major), each a sigmoid saliency probability.
void nn_saliency(const float* scene, float* sal /*[N_TILES]*/);

// Stage 3a: connected components (4-neighbour) of (sal >= REGION_THR).
// Fills out[] (sorted largest-first), returns region count (<= maxr).
int nn_regions(const float* sal, Region* out, int maxr);

// Stage 3b: regions -> crop windows per REGION_MODE. Returns window count.
int nn_windows(const float* sal, const Region* regs, int nreg,
               Window* out, int maxw);

// Bilinear-resize a scene sub-rectangle into a size*size buffer (matches
// data_common.crop / sim_pipeline.bilinear_resize).
void nn_crop_resize(const float* scene, int x0, int y0, int w, int h,
                    float* dst, int size);

// Stage 4: crop[CROP*CROP] -> class probs[N_CLASS] + decoded box[4]=(cx,cy,w,h).
void nn_bnn(const float* crop, float* probs /*[N_CLASS]*/, float* box /*[4]*/);
