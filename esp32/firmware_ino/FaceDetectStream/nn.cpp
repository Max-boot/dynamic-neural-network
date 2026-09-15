// ============================================================================
//  nn.cpp  -  Face-cascade forward pass. Mirrors sim_pipeline.py.
//
//  Precision note: the host reference accumulates convolutions in float64.
//  Here we accumulate in float32 (the ESP32 has no hardware double). This stays
//  well inside the verified BNN tolerance (argmax + <0.15 prob drift) and does
//  not change region/box results in practice; the saliency MLP head (12->16->1)
//  accumulates in double to keep the map faithful.
// ============================================================================
#include "nn.h"
#include "model.h"
#include <math.h>

// ---- PSRAM scratch ---------------------------------------------------------
static float* s_f1 = nullptr;   // [8,128,128]
static float* s_f2 = nullptr;   // [4,128,128]
static float* b_f1 = nullptr;   // [BNN_C1,28,28]
static float* b_p1 = nullptr;   // [BNN_C1,14,14]
static float* b_f2 = nullptr;   // [BNN_C2,14,14]
static float* b_p2 = nullptr;   // [BNN_C2,7,7]

static float* palloc(size_t n) {
  float* p = (float*)heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!p) p = (float*)malloc(n * sizeof(float));
  return p;
}

bool nn_begin() {
  s_f1 = palloc(8 * SCENE * SCENE);
  s_f2 = palloc(4 * SCENE * SCENE);
  b_f1 = palloc(BNN_C1 * CROP * CROP);
  b_p1 = palloc(BNN_C1 * 14 * 14);
  b_f2 = palloc(BNN_C2 * 14 * 14);
  b_p2 = palloc(BNN_C2 * 7 * 7);
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

// ---- bilinear resize of a scene sub-rectangle (all N_CH planes) -------------
void nn_crop_resize(const float* scene, int x0, int y0, int w, int h,
                    float* dst, int size) {
  // linspace(0, dim-1, size) sampling, matches data_common.crop.
  const float sy = (h > 1) ? (float)(h - 1) / (float)(size - 1) : 0.0f;
  const float sx = (w > 1) ? (float)(w - 1) / (float)(size - 1) : 0.0f;
  for (int c = 0; c < N_CH; c++) {
    const float* sc = scene + c * SCENE * SCENE;   // channel plane
    float* dc = dst + c * size * size;
    for (int j = 0; j < size; j++) {
      float fy = j * sy;
      int yy0 = (int)floorf(fy);
      int yy1 = yy0 + 1; if (yy1 > h - 1) yy1 = h - 1;
      float wy = fy - yy0;
      const float* r0 = sc + (y0 + yy0) * SCENE + x0;
      const float* r1 = sc + (y0 + yy1) * SCENE + x0;
      for (int i = 0; i < size; i++) {
        float fx = i * sx;
        int xx0 = (int)floorf(fx);
        int xx1 = xx0 + 1; if (xx1 > w - 1) xx1 = w - 1;
        float wx = fx - xx0;
        float v = r0[xx0] * (1 - wy) * (1 - wx)
                + r0[xx1] * (1 - wy) * wx
                + r1[xx0] * wy * (1 - wx)
                + r1[xx1] * wy * wx;
        dc[j * size + i] = v;
      }
    }
  }
}

// ---- Stage 1+2: saliency ---------------------------------------------------
void nn_saliency(const float* scene, float* sal) {
  conv3x3_same_relu(scene, N_CH, SCENE, SCENE, g_sal.c1w, g_sal.c1b, 8, s_f1);
  conv3x3_same_relu(s_f1, 8, SCENE, SCENE, g_sal.c2w, g_sal.c2b, 4, s_f2);

  // Per-channel tile stats: for each of the 64 tiles and each of the 4 conv
  // channels, mean / max / population-var (ddof=0) over the 16x16 = 256 values.
  // Feature layout per tile: feat[t][c*3 + {0:mean, 1:max, 2:var}] -> 12 values.
  // feat is 64*12*4 = 3072 B on the stack; nn_saliency runs on inference_task
  // (32 KB stack) -- safe. Indices are all fixed [N_TILES][12] -> no OOB.
  float feat[N_TILES][12];
  for (int j = 0; j < GRID; j++) {
    for (int i = 0; i < GRID; i++) {
      int t = j * GRID + i;
      for (int c = 0; c < 4; c++) {
        double sum = 0.0, sq = 0.0;
        float m = -3.4e38f;
        for (int ty = 0; ty < TILE; ty++) {
          const float* row = s_f2 + (c * SCENE + (j * TILE + ty)) * SCENE + i * TILE;
          for (int tx = 0; tx < TILE; tx++) {
            float v = row[tx];
            sum += v; sq += (double)v * v;
            if (v > m) m = v;
          }
        }
        double n = (double)(TILE * TILE);         // 256
        double mu = sum / n;
        double vr = sq / n - mu * mu;             // population variance (ddof=0)
        if (vr < 0) vr = 0;
        feat[t][c * 3 + 0] = (float)mu;
        feat[t][c * 3 + 1] = m;
        feat[t][c * 3 + 2] = (float)vr;
      }
    }
  }

  // Instance-norm each of the 12 feature columns across the 64 tiles
  // (sample std, ddof=1, + 1e-5, clamp +-3).
  for (int k = 0; k < 12; k++) {
    double sm = 0.0;
    for (int t = 0; t < N_TILES; t++) sm += feat[t][k];
    double avg = sm / N_TILES;
    double ss = 0.0;
    for (int t = 0; t < N_TILES; t++) { double d = feat[t][k] - avg; ss += d * d; }
    double sd = sqrt(ss / (N_TILES - 1));         // ddof=1
    double denom = sd + 1e-5;
    for (int t = 0; t < N_TILES; t++) {
      double z = (feat[t][k] - avg) / denom;
      if (z > 3.0) z = 3.0; else if (z < -3.0) z = -3.0;
      feat[t][k] = (float)z;
    }
  }

  // MLP head 12 -> 16 -> 1: h = relu(fc1w @ x + fc1b); y = fc2w @ h + fc2b.
  // Double accumulation, o (output) outer / k (input) inner, then sigmoid.
  const float* fc1w = g_sal.fc1w;    // (16,12)
  const float* fc1b = g_sal.fc1b;    // (16)
  const float* fc2w = g_sal.fc2w;    // (1,16)
  const float* fc2b = g_sal.fc2b;    // (1)
  for (int t = 0; t < N_TILES; t++) {
    const float* xv = feat[t];       // [12]
    float h[16];
    for (int o = 0; o < 16; o++) {
      double acc = fc1b[o];
      const float* wrow = fc1w + o * 12;
      for (int k = 0; k < 12; k++) acc += (double)wrow[k] * xv[k];
      h[o] = acc > 0.0 ? (float)acc : 0.0f;       // ReLU
    }
    double y = fc2b[0];
    for (int o = 0; o < 16; o++) y += (double)fc2w[o] * h[o];
    sal[t] = (float)(1.0 / (1.0 + exp(-y)));
  }
}

// ---- Stage 3a: proposal regions from the saliency map ----------------------
static inline int iabs_(int v) { return v < 0 ? -v : v; }

// Union-find with path halving (deterministic, no recursion). parent[] is a plain
// int array sized N_TILES; returns the representative tile of x's set.
static int uf_find(int* parent, int x) {
  while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
  return x;
}

#if REGION_MODE == REGION_MODE_ADAPTIVE
// Adaptive hysteresis regions + valley-based peak split. Mirrors
// sim_pipeline.py _adaptive_regions() 1:1. All scratch is fixed [N_TILES], so the
// output count can never exceed N_TILES (each tile lands in at most one region).
int nn_regions(const float* sal, Region* out, int maxr) {
  // 1. scene statistics over the 64 tiles (mean + sample std, ddof=1).
  double sum = 0.0;
  for (int t = 0; t < N_TILES; t++) sum += sal[t];
  double mu = sum / N_TILES;
  double ss = 0.0;
  for (int t = 0; t < N_TILES; t++) { double d = sal[t] - mu; ss += d * d; }
  double sd = sqrt(ss / (double)(N_TILES - 1));       // ddof=1

  // 2. hysteresis thresholds (clamped so flat/empty scenes seed nothing and a
  //    very confident face always seeds).
  double th = mu + (double)SEED_K * sd;
  if (th < (double)SEED_FLOOR) th = SEED_FLOOR;
  if (th > (double)SEED_CEIL)  th = SEED_CEIL;
  double tl = mu + (double)GROW_K * sd;
  if (tl < (double)GROW_FLOOR) tl = GROW_FLOOR;
  if (tl > th)                 tl = th;
  const float thf = (float)th, tlf = (float)tl;

  // 3. hysteresis connected components: BFS starts ONLY at seed tiles (>=T_high),
  //    grows across 4-neighbours that clear the grow threshold (>=T_low).
  int comp[N_TILES];
  bool seen[N_TILES];
  for (int t = 0; t < N_TILES; t++) { comp[t] = -1; seen[t] = false; }
  int stack[N_TILES];
  const int dy4[4] = { -1, 1, 0, 0 }, dx4[4] = { 0, 0, -1, 1 };
  int ncomp = 0;
  for (int start = 0; start < N_TILES; start++) {
    if (seen[start] || sal[start] < thf) continue;    // only seeds open a comp
    int sp = 0;
    stack[sp++] = start; seen[start] = true;
    while (sp > 0) {
      int t = stack[--sp];
      comp[t] = ncomp;
      int y = t / GRID, x = t % GRID;
      for (int d = 0; d < 4; d++) {
        int ny = y + dy4[d], nx = x + dx4[d];
        if (ny < 0 || ny >= GRID || nx < 0 || nx >= GRID) continue;
        int nt = ny * GRID + nx;
        if (!seen[nt] && sal[nt] >= tlf) { seen[nt] = true; stack[sp++] = nt; }
      }
    }
    ncomp++;
  }
  if (ncomp == 0) return 0;

  // group accumulators (reused per component; sized for the worst case).
  int gi0[N_TILES], gj0[N_TILES], gi1[N_TILES], gj1[N_TILES], gcnt[N_TILES];
  double gscore[N_TILES];
  int acc[N_TILES];
  // watershed scratch (reused per component).
  int ord[N_TILES], posrank[N_TILES], parent[N_TILES], peaktile[N_TILES];
  bool ispeak[N_TILES];
  float prom[N_TILES];

  int nout = 0;
  for (int c = 0; c < ncomp && nout < maxr; c++) {
    // 4. prominence-based peak split (watershed by union-find). Process the
    //    component's tiles in descending saliency; the first tile of each still
    //    disconnected high spot opens a peak, and when a lower tile finally joins
    //    two spots that saddle level fixes the weaker peak's prominence (its
    //    height above the saddle). A peak survives only if prominence >=
    //    PROMINENCE, so a flat plateau (saddle == peak -> prominence 0) stays ONE
    //    peak while two hills across a real valley both survive.
    int nord = 0;
    for (int t = 0; t < N_TILES; t++) if (comp[t] == c) ord[nord++] = t;
    for (int a = 1; a < nord; a++) {          // sort desc by sal (tie: tile asc)
      int key = ord[a], b = a - 1;
      while (b >= 0 && (sal[ord[b]] < sal[key] ||
             (sal[ord[b]] == sal[key] && ord[b] > key))) { ord[b + 1] = ord[b]; b--; }
      ord[b + 1] = key;
    }
    for (int t = 0; t < N_TILES; t++) {
      posrank[t] = N_TILES + 1; parent[t] = t; peaktile[t] = t;
      ispeak[t] = false; prom[t] = 3.4e38f;
    }
    for (int a = 0; a < nord; a++) posrank[ord[a]] = a;
    for (int a = 0; a < nord; a++) {
      int t = ord[a];
      int y = t / GRID, x = t % GRID;
      int roots[4], nroots = 0;             // distinct sets of already-seen neighbours
      for (int d = 0; d < 4; d++) {
        int ny = y + dy4[d], nx = x + dx4[d];
        if (ny < 0 || ny >= GRID || nx < 0 || nx >= GRID) continue;
        int nt = ny * GRID + nx;
        if (comp[nt] != c || posrank[nt] >= a) continue;   // only higher/earlier tiles
        int rt = uf_find(parent, nt);
        bool dup = false;
        for (int q = 0; q < nroots; q++) if (roots[q] == rt) { dup = true; break; }
        if (!dup && nroots < 4) roots[nroots++] = rt;
      }
      if (nroots == 0) {
        ispeak[t] = true;                   // opens a new peak
      } else {
        int hi = roots[0];                  // strongest neighbour peak survives
        for (int q = 1; q < nroots; q++) {
          int pa = peaktile[roots[q]], pb = peaktile[hi];
          if (sal[pa] > sal[pb] || (sal[pa] == sal[pb] && pa < pb)) hi = roots[q];
        }
        int hipeak = peaktile[hi];
        for (int q = 0; q < nroots; q++) {  // every weaker peak dies at this saddle
          if (roots[q] == hi) continue;
          int lp = peaktile[roots[q]];
          float pr = sal[lp] - sal[t];
          if (pr < prom[lp]) prom[lp] = pr;
        }
        parent[t] = hi;
        for (int q = 0; q < nroots; q++) if (roots[q] != hi) parent[roots[q]] = hi;
        peaktile[hi] = hipeak;
      }
    }
    // accepted peaks: prominence >= PROMINENCE (the surviving top peak keeps INF).
    int nacc = 0;
    for (int t = 0; t < N_TILES; t++)
      if (comp[t] == c && ispeak[t] && prom[t] >= (float)PROMINENCE) acc[nacc++] = t;
    // order accepted peaks strongest-first (deterministic nearest-peak tie-break).
    for (int a = 1; a < nacc; a++) {
      int key = acc[a], b = a - 1;
      while (b >= 0 && (sal[acc[b]] < sal[key] ||
             (sal[acc[b]] == sal[key] && acc[b] > key))) { acc[b + 1] = acc[b]; b--; }
      acc[b + 1] = key;
    }
    // (a component opened from a seed always has >=1 surviving peak: its global
    //  maximum never dies, so prom == INF >= PROMINENCE and nacc >= 1.)

    // 5. assign each comp tile to its nearest accepted peak (Manhattan; tie ->
    //    lower accepted index). nacc<=1 -> the whole component is one region.
    int ngroup = (nacc <= 1) ? 1 : nacc;
    for (int g = 0; g < ngroup; g++) {
      gi0[g] = GRID; gj0[g] = GRID; gi1[g] = -1; gj1[g] = -1;
      gcnt[g] = 0; gscore[g] = 0.0;
    }
    for (int t = 0; t < N_TILES; t++) {
      if (comp[t] != c) continue;
      int y = t / GRID, x = t % GRID;
      int g = 0;
      if (nacc > 1) {
        int best = 0, bestd = 1 << 30;
        for (int b = 0; b < nacc; b++) {
          int ay = acc[b] / GRID, ax = acc[b] % GRID;
          int md = iabs_(y - ay) + iabs_(x - ax);
          if (md < bestd) { bestd = md; best = b; }
        }
        g = best;
      }
      if (x < gi0[g]) gi0[g] = x; if (x > gi1[g]) gi1[g] = x;
      if (y < gj0[g]) gj0[g] = y; if (y > gj1[g]) gj1[g] = y;
      gcnt[g]++; gscore[g] += sal[t];
    }
    for (int g = 0; g < ngroup && nout < maxr; g++) {
      if (gcnt[g] == 0) continue;
      out[nout].x0 = gi0[g] * TILE;
      out[nout].y0 = gj0[g] * TILE;
      out[nout].x1 = (gi1[g] + 1) * TILE;
      out[nout].y1 = (gj1[g] + 1) * TILE;
      out[nout].tiles = gcnt[g];
      out[nout].score = (float)gscore[g];
      nout++;
    }
  }

  // Sort by saliency score, strongest first (insertion sort; nout is tiny). The
  // window cap in nn_windows then keeps the strongest proposals.
  for (int i = 1; i < nout; i++) {
    Region key = out[i];
    int j = i - 1;
    while (j >= 0 && out[j].score < key.score) { out[j + 1] = out[j]; j--; }
    out[j + 1] = key;
  }
  return nout;
}
#else
// Legacy: connected components (4-neighbour BFS) of (sal >= REGION_THR).
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
    out[n].score = (float)nc;
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
#endif

// ---- Stage 3b: region -> window(s) -----------------------------------------
#if REGION_MODE == REGION_MODE_ADAPTIVE
// Rectangular, region-proportional windows. Mirrors sim_pipeline.py
// _adaptive_windows() 1:1: %margin (integer), min-side floor, clip to scene.
int nn_windows(const float* sal, const Region* regs, int nreg,
               Window* out, int maxw) {
  (void)sal;                                  // windows come from region extent
  int nw = 0;
  for (int r = 0; r < nreg && nw < maxw; r++) {
    int x0 = regs[r].x0, y0 = regs[r].y0, x1 = regs[r].x1, y1 = regs[r].y1;
    // proportional margin (integer math -> bit-identical to the sim's //).
    int mw = (x1 - x0) * MARGIN_PCT / 100;
    int mh = (y1 - y0) * MARGIN_PCT / 100;
    x0 -= mw; x1 += mw; y0 -= mh; y1 += mh;
    // floor each side to WIN_FLOOR, expanding symmetrically around the centre.
    int wd = x1 - x0;
    if (wd < WIN_FLOOR) { int need = WIN_FLOOR - wd; x0 -= need / 2; x1 += need - need / 2; }
    int hd = y1 - y0;
    if (hd < WIN_FLOOR) { int need = WIN_FLOOR - hd; y0 -= need / 2; y1 += need - need / 2; }
    // clip to the scene (a floored side may shrink again near an edge).
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 > SCENE) x1 = SCENE; if (y1 > SCENE) y1 = SCENE;
    if (x1 - x0 < 4 || y1 - y0 < 4) continue;
    out[nw].x0 = x0; out[nw].y0 = y0; out[nw].x1 = x1; out[nw].y1 = y1;
    nw++;
  }
  return nw;
}
#else
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
#endif

// ---- Stage 4: BNN (int8 FC1, deterministic) --------------------------------
void nn_bnn(const float* crop, float* probs, float* box) {
  conv3x3_same_relu(crop, N_CH, CROP, CROP, g_bnn.b1w, g_bnn.b1b, BNN_C1, b_f1);
  maxpool2(b_f1, BNN_C1, CROP, CROP, b_p1);                 // -> [BNN_C1,14,14]
  conv3x3_same_relu(b_p1, BNN_C1, 14, 14, g_bnn.b2w, g_bnn.b2b, BNN_C2, b_f2);
  maxpool2(b_f2, BNN_C2, 14, 14, b_p2);                     // -> [BNN_C2,7,7]
  const float* flat = b_p2;                             // BNN_FLAT, C-order

  // FC1 (int8): h[r] = relu( sum_k flat[k]*W8[r][k] * fc1s[r] + fc1b[r] ).
  float h[BNN_HIDDEN];
  for (int r = 0; r < BNN_HIDDEN; r++) {
    const int8_t* wr = g_bnn.fc1w8 + (size_t)r * BNN_FLAT;
    float acc = 0.0f;
    for (int k = 0; k < BNN_FLAT; k++) acc += flat[k] * (float)wr[k];
    float v = acc * g_bnn.fc1s[r] + g_bnn.fc1b[r];
    h[r] = v > 0.0f ? v : 0.0f;
  }

  // FC2 -> logits -> softmax.
  float logit[N_CLASS];
  for (int o = 0; o < N_CLASS; o++) {
    const float* wr = g_bnn.fc2w + o * BNN_HIDDEN;
    float acc = g_bnn.fc2b[o];
    for (int k = 0; k < BNN_HIDDEN; k++) acc += wr[k] * h[k];
    logit[o] = acc;
  }
  float mx = logit[0];
  for (int o = 1; o < N_CLASS; o++) if (logit[o] > mx) mx = logit[o];
  float sum = 0.0f;
  for (int o = 0; o < N_CLASS; o++) { probs[o] = expf(logit[o] - mx); sum += probs[o]; }
  for (int o = 0; o < N_CLASS; o++) probs[o] /= sum;

  // Student BNN: no box head (fc3) -- the box is the adaptive saliency window
  // itself. Normalized coords: center (0.5,0.5), size 1x1 of the window.
  box[0] = 0.5f; box[1] = 0.5f; box[2] = 1.0f; box[3] = 1.0f;
}
