// ============================================================================
//  nn.h  -  Forward-pass primitives for the face cascade. Every routine mirrors
//           esp32/tools/sim_pipeline.py so the firmware can be audited line by
//           line against that (verified) NumPy reference.
// ============================================================================
#pragma once
#include <Arduino.h>
#include "config.h"

struct Region  { int x0, y0, x1, y1, tiles; float score; };  // score = sum of tile saliency
struct Window  { int x0, y0, x1, y1; };
struct Detection { float x0, y0, x1, y1; int cls; float conf; };

// Allocate all PSRAM scratch buffers. Call once at boot AFTER psramInit().
bool nn_begin();

// Stage 1+2: scene[N_CH*SCENE*SCENE] (channel-major RGB float 0..1, planes R,G,B)
// -> sal[N_TILES] (8x8, row-major), each a sigmoid saliency probability.
// Implements the model selected by SALIENCY_MODEL in config.h (MLP head or
// linear-bottleneck conv net); both mirror their host reference exactly.
void nn_saliency(const float* scene, float* sal /*[N_TILES]*/);

// Stage 3a: proposal regions from the saliency map. REGION_MODE_ADAPTIVE uses
// adaptive hysteresis (per-frame T_high/T_low) + valley-based peak splitting, so
// one big face stays one region while two faces across a valley split. Legacy
// modes use fixed-threshold (REGION_THR) connected components. Regions are sorted
// by saliency (strongest first); returns the region count (<= maxr).
int nn_regions(const float* sal, Region* out, int maxr);

// Stage 3b: regions -> crop windows per REGION_MODE (rectangular & region-sized
// under REGION_MODE_ADAPTIVE). Returns window count (<= maxw).
int nn_windows(const float* sal, const Region* regs, int nreg,
               Window* out, int maxw);

// Bilinear-resize a scene sub-rectangle (all N_CH planes) into a size*size*N_CH
// buffer (channel-major; matches data_common.crop / sim_pipeline.bilinear_resize).
void nn_crop_resize(const float* scene, int x0, int y0, int w, int h,
                    float* dst, int size);

// Stage 4: crop[N_CH*CROP*CROP] -> class probs[N_CLASS] + decoded box[4]=(cx,cy,w,h).
void nn_bnn(const float* crop, float* probs /*[N_CLASS]*/, float* box /*[4]*/);
