// ============================================================================
//  nn.cpp  -  Face-cascade forward pass. Mirrors sim_pipeline.py.
//
//  Precision note: the host reference accumulates convolutions in float64.
//  Here we accumulate in float32 (the ESP32 has no hardware double). This stays
//  well inside the verified BNN tolerance (argmax + <0.15 prob drift) and does
//  not change region/box results in practice; the small-signal ANFIS reduction
//  is done carefully to keep the saliency map faithful.
// ============================================================================
#include "nn.h"
#include "model.h"
#include <math.h>

// ---- PSRAM scratch ---------------------------------------------------------
static float* s_f1 = nullptr;   // [8,128,128]
static float* s_f2 = nullptr;   // [4,128,128]
static float* b_f1 = nullptr;   // [40,28,28]
static float* b_p1 = nullptr;   // [40,14,14]
static float* b_f2 = nullptr;   // [80,14,14]
static float* b_p2 = nullptr;   // [80,7,7]

static float* palloc(size_t n) {
  float* p = (float*)heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!p) p = (float*)malloc(n * sizeof(float));
  return p;
}

bool nn_begin() {
  s_f1 = palloc(8 * SCENE * SCENE);
  s_f2 = palloc(4 * SCENE * SCENE);
  b_f1 = palloc(40 * CROP * CROP);
  b_p1 = palloc(40 * 14 * 14);
  b_f2 = palloc(80 * 14 * 14);
  b_p2 = palloc(80 * 7 * 7);
  bool ok = s_f1 && s_f2 && b_f1 && b_p1 && b_f2 && b_p2;
  if (!ok) Serial.println("[nn] scratch alloc failed");
  return ok;
}

// ---- generic conv/pool -----------------------------------------------------
// 3x3 same-pad conv + ReLU. in[C,H,W], w[O,C,3,3], b[O] -> out[O,H,W].
static void conv3x3_same_relu(const float* in, int C, int H, int W,
                              const float* w, const float* b, int O,
                              float* out) {
  for (int o = 0; o < O; o++) {
    const float bias = b[o];
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        float acc = bias;
        for (int i = 0; i < C; i++) {
          const float* wp = w + ((o * C + i) * 9);   // 3x3 block for (o,i)
          const float* ip = in + (i * H) * W;
          for (int dy = -1; dy <= 1; dy++) {
            int yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            const float* row = ip + yy * W;
            const float* wr = wp + (dy + 1) * 3 + 1;  // center of this row
            for (int dx = -1; dx <= 1; dx++) {
              int xx = x + dx;
              if (xx < 0 || xx >= W) continue;
              acc += wr[dx] * row[xx];
            }
          }
        }
        out[(o * H + y) * W + x] = acc > 0.0f ? acc : 0.0f;
      }
    }
  }
}

// 2x2 max-pool. in[C,H,W] -> out[C,H/2,W/2].
static void maxpool2(const float* in, int C, int H, int W, float* out) {
  int OH = H / 2, OW = W / 2;
  for (int c = 0; c < C; c++) {
    for (int y = 0; y < OH; y++) {
      for (int x = 0; x < OW; x++) {
        const float* p = in + (c * H + 2 * y) * W + 2 * x;
        float m = p[0];
        if (p[1] > m) m = p[1];
        if (p[W] > m) m = p[W];
        if (p[W + 1] > m) m = p[W + 1];
        out[(c * OH + y) * OW + x] = m;
      }
    }
  }
}

// ---- bilinear resize of a scene sub-rectangle ------------------------------
void nn_crop_resize(const float* scene, int x0, int y0, int w, int h,
                    float* dst, int size) {
  // linspace(0, dim-1, size) sampling, matches data_common.crop.
  const float sy = (h > 1) ? (float)(h - 1) / (float)(size - 1) : 0.0f;
  const float sx = (w > 1) ? (float)(w - 1) / (float)(size - 1) : 0.0f;
  for (int j = 0; j < size; j++) {
    float fy = j * sy;
    int yy0 = (int)floorf(fy);
    int yy1 = yy0 + 1; if (yy1 > h - 1) yy1 = h - 1;
    float wy = fy - yy0;
    const float* r0 = scene + (y0 + yy0) * SCENE + x0;
    const float* r1 = scene + (y0 + yy1) * SCENE + x0;
    for (int i = 0; i < size; i++) {
      float fx = i * sx;
      int xx0 = (int)floorf(fx);
      int xx1 = xx0 + 1; if (xx1 > w - 1) xx1 = w - 1;
      float wx = fx - xx0;
      float v = r0[xx0] * (1 - wy) * (1 - wx)
              + r0[xx1] * (1 - wy) * wx
              + r1[xx0] * wy * (1 - wx)
              + r1[xx1] * wy * wx;
      dst[j * size + i] = v;
    }
  }
}

// ---- Stage 1+2: saliency ---------------------------------------------------
void nn_saliency(const float* scene, float* sal) {
  conv3x3_same_relu(scene, 1, SCENE, SCENE, g_sal.c1w, g_sal.c1b, 8, s_f1);
  conv3x3_same_relu(s_f1, 8, SCENE, SCENE, g_sal.c2w, g_sal.c2b, 4, s_f2);

  // Per-tile mean / max / population-var over the 4x16x16 = 1024 values.
  float mean[N_TILES], mx[N_TILES], var[N_TILES];
  for (int j = 0; j < GRID; j++) {
    for (int i = 0; i < GRID; i++) {
      double sum = 0.0, sq = 0.0;
      float m = -3.4e38f;
      for (int c = 0; c < 4; c++) {
        for (int ty = 0; ty < TILE; ty++) {
          const float* row = s_f2 + (c * SCENE + (j * TILE + ty)) * SCENE + i * TILE;
          for (int tx = 0; tx < TILE; tx++) {
            float v = row[tx];
            sum += v; sq += (double)v * v;
            if (v > m) m = v;
          }
        }
      }
      double n = 4.0 * TILE * TILE;             // 1024
      double mu = sum / n;
      double vr = sq / n - mu * mu;             // population variance (ddof=0)
      if (vr < 0) vr = 0;
      int t = j * GRID + i;
      mean[t] = (float)mu; mx[t] = m; var[t] = (float)vr;
    }
  }

  // Instance-norm each statistic across the 64 tiles (sample std, ddof=1).
  float* stat[3] = { mean, mx, var };
  for (int k = 0; k < 3; k++) {
    float* s = stat[k];
    double sm = 0.0;
    for (int t = 0; t < N_TILES; t++) sm += s[t];
    double avg = sm / N_TILES;
    double ss = 0.0;
    for (int t = 0; t < N_TILES; t++) { double d = s[t] - avg; ss += d * d; }
    double sd = sqrt(ss / (N_TILES - 1));       // ddof=1
    double denom = sd + 1e-5;
    for (int t = 0; t < N_TILES; t++) {
      double z = (s[t] - avg) / denom;
      if (z > 3.0) z = 3.0; else if (z < -3.0) z = -3.0;
      s[t] = (float)z;
    }
  }

  // ANFIS: 125 rules (5 MF ^ 3 inputs). xv = [mean, max, var].
  const float* c = g_sal.c;          // (3,5)
  const float* sg = g_sal.sigma;     // (3,5) softplus
  const float* P = g_sal.P;          // (125,4)
  for (int t = 0; t < N_TILES; t++) {
    float xv[3] = { mean[t], mx[t], var[t] };
    double wsum = 0.0, wlin = 0.0;
    for (int r = 0; r < 125; r++) {
      int m0 = (r / 25) % 5, m1 = (r / 5) % 5, m2 = r % 5;
      int idx[3] = { m0, m1, m2 };
      double val = 1.0;
      for (int k = 0; k < 3; k++) {
        float ck = c[k * 5 + idx[k]];
        float sk = sg[k * 5 + idx[k]];
        double dk = xv[k] - ck;
        val *= exp(-(dk * dk) / (2.0 * (double)sk * sk));
      }
      const float* pr = P + r * 4;
      double lin = pr[0] + pr[1] * xv[0] + pr[2] * xv[1] + pr[3] * xv[2];
      wsum += val;
      wlin += val * lin;
    }
    double y = wlin / (wsum + 1e-6);
    sal[t] = (float)(1.0 / (1.0 + exp(-y)));
  }
}

// ---- Stage 3a: connected components (4-neighbour BFS) ----------------------
int nn_regions(const float* sal, Region* out, int maxr) {
  bool active[N_TILES], seen[N_TILES];
  for (int t = 0; t < N_TILES; t++) { active[t] = sal[t] >= REGION_THR; seen[t] = false; }
  int stack[N_TILES], cells[N_TILES];
  int n = 0;
  for (int start = 0; start < N_TILES && n < maxr; start++) {
    if (!active[start] || seen[start]) continue;
    int sp = 0, nc = 0;
    stack[sp++] = start; seen[start] = true;
    int minx = GRID, miny = GRID, maxx = -1, maxy = -1;
    while (sp > 0) {
      int t = stack[--sp];
      cells[nc++] = t;
      int y = t / GRID, x = t % GRID;
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
      const int dy[4] = { -1, 1, 0, 0 }, dx[4] = { 0, 0, -1, 1 };
      for (int d = 0; d < 4; d++) {
        int ny = y + dy[d], nx = x + dx[d];
        if (ny < 0 || ny >= GRID || nx < 0 || nx >= GRID) continue;
        int nt = ny * GRID + nx;
        if (active[nt] && !seen[nt]) { seen[nt] = true; stack[sp++] = nt; }
      }
    }
    out[n].x0 = minx * TILE; out[n].y0 = miny * TILE;
    out[n].x1 = (maxx + 1) * TILE; out[n].y1 = (maxy + 1) * TILE;
    out[n].tiles = nc;
    n++;
  }
  // Sort largest-first (insertion sort; n is tiny).
  for (int i = 1; i < n; i++) {
    Region key = out[i];
    int j = i - 1;
    while (j >= 0 && out[j].tiles < key.tiles) { out[j + 1] = out[j]; j--; }
    out[j + 1] = key;
  }
  return n;
}

// ---- Stage 3b: region -> window(s) -----------------------------------------
static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int nn_windows(const float* sal, const Region* regs, int nreg,
               Window* out, int maxw) {
  int nw = 0;
#if REGION_MODE == REGION_MODE_BBOX
  for (int r = 0; r < nreg && nw < maxw; r++) {
    int x0 = regs[r].x0 - BBOX_MARGIN, y0 = regs[r].y0 - BBOX_MARGIN;
    int x1 = regs[r].x1 + BBOX_MARGIN, y1 = regs[r].y1 + BBOX_MARGIN;
    x0 = clampi(x0, 0, SCENE); y0 = clampi(y0, 0, SCENE);
    x1 = clampi(x1, 0, SCENE); y1 = clampi(y1, 0, SCENE);
    if (x1 - x0 < 4 || y1 - y0 < 4) continue;
    out[nw++] = { x0, y0, x1, y1 };
  }
#else  // REGION_MODE_REFOCUS
  for (int r = 0; r < nreg && nw < maxw; r++) {
    int x0 = regs[r].x0, y0 = regs[r].y0, x1 = regs[r].x1, y1 = regs[r].y1;
    int i0 = x0 / TILE, j0 = y0 / TILE;
    int i1 = (x1 - 1) / TILE, j1 = (y1 - 1) / TILE;
    // Saliency-weighted centroid within the region's tile block.
    double wsum = 0.0, cxs = 0.0, cys = 0.0;
    for (int jj = j0; jj <= j1; jj++) {
      for (int ii = i0; ii <= i1; ii++) {
        double sv = sal[jj * GRID + ii];
        wsum += sv;
        cxs += sv * (ii - i0);      // local col index (matches numpy.indices)
        cys += sv * (jj - j0);      // local row index
      }
    }
    if (wsum <= 0.0) continue;
    double cx_t = cxs / wsum, cy_t = cys / wsum;
    double cx = x0 + (cx_t + 0.5) * TILE;
    double cy = y0 + (cy_t + 0.5) * TILE;
    int extent = (x1 - x0) > (y1 - y0) ? (x1 - x0) : (y1 - y0);
    int w = extent; if (w < WIN_MIN) w = WIN_MIN; if (w > WIN_MAX) w = WIN_MAX;
    // int(clip(cx - w/2, 0, 128 - w)) -- clip first, then truncate (>=0).
    double xf = cx - w / 2.0; if (xf < 0) xf = 0; if (xf > SCENE - w) xf = SCENE - w;
    double yf = cy - w / 2.0; if (yf < 0) yf = 0; if (yf > SCENE - w) yf = SCENE - w;
    int xc0 = (int)xf, yc0 = (int)yf;
    out[nw++] = { xc0, yc0, xc0 + w, yc0 + w };
  }
#endif
  return nw;
}

// ---- Stage 4: BNN (int8 FC1, deterministic) --------------------------------
void nn_bnn(const float* crop, float* probs, float* box) {
  conv3x3_same_relu(crop, 1, CROP, CROP, g_bnn.b1w, g_bnn.b1b, 40, b_f1);
  maxpool2(b_f1, 40, CROP, CROP, b_p1);                 // -> [40,14,14]
  conv3x3_same_relu(b_p1, 40, 14, 14, g_bnn.b2w, g_bnn.b2b, 80, b_f2);
  maxpool2(b_f2, 80, 14, 14, b_p2);                     // -> [80,7,7]
  const float* flat = b_p2;                             // 3920, C-order

  // FC1 (int8): h[r] = relu( sum_k flat[k]*W8[r][k] * fc1s[r] + fc1b[r] ).
  float h[192];
  for (int r = 0; r < 192; r++) {
    const int8_t* wr = g_bnn.fc1w8 + (size_t)r * BNN_FLAT;
    float acc = 0.0f;
    for (int k = 0; k < BNN_FLAT; k++) acc += flat[k] * (float)wr[k];
    float v = acc * g_bnn.fc1s[r] + g_bnn.fc1b[r];
    h[r] = v > 0.0f ? v : 0.0f;
  }

  // FC2 -> logits -> softmax.
  float logit[N_CLASS];
  for (int o = 0; o < N_CLASS; o++) {
    const float* wr = g_bnn.fc2w + o * 192;
    float acc = g_bnn.fc2b[o];
    for (int k = 0; k < 192; k++) acc += wr[k] * h[k];
    logit[o] = acc;
  }
  float mx = logit[0];
  for (int o = 1; o < N_CLASS; o++) if (logit[o] > mx) mx = logit[o];
  float sum = 0.0f;
  for (int o = 0; o < N_CLASS; o++) { probs[o] = expf(logit[o] - mx); sum += probs[o]; }
  for (int o = 0; o < N_CLASS; o++) probs[o] /= sum;

  // FC3 -> raw box -> decode (cx=sig, cy=sig, w=exp, h=exp).
  float raw[4];
  for (int o = 0; o < 4; o++) {
    const float* wr = g_bnn.fc3w + o * 192;
    float acc = g_bnn.fc3b[o];
    for (int k = 0; k < 192; k++) acc += wr[k] * h[k];
    raw[o] = acc;
  }
  box[0] = 1.0f / (1.0f + expf(-raw[0]));
  box[1] = 1.0f / (1.0f + expf(-raw[1]));
  box[2] = expf(raw[2]);
  box[3] = expf(raw[3]);
}
