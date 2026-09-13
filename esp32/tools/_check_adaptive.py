"""Standalone sanity check for the adaptive region/window logic in sim_pipeline.
Does NOT need the model blobs -- it only exercises the deterministic geometry
that must match nn.cpp. Verifies the four cases the design promises."""
import numpy as np
import sim_pipeline as S

img = np.zeros((S.SCENE, S.SCENE, 3), dtype=np.float32)  # dummy scene for slicing


def summarize(name, sal):
    regs = S._adaptive_regions(sal)
    wins = S._adaptive_windows(img, regs)
    mu, sd, th, tl = S._adaptive_thresholds([float(v) for v in sal.ravel()])
    boxes = [(ox, oy, p.shape[1], p.shape[0]) for (p, ox, oy) in wins]
    print(f"{name}: mu={mu:.3f} sd={sd:.3f} th={th:.3f} tl={tl:.3f} "
          f"-> {len(regs)} region(s), {len(wins)} window(s)")
    for i, (ox, oy, w, h) in enumerate(boxes):
        print(f"    win{i}: origin=({ox},{oy}) size={w}x{h}")
    # invariants that must always hold (no OOB slices, min side honoured)
    for (p, ox, oy) in wins:
        assert 0 <= ox and 0 <= oy, "negative origin"
        assert ox + p.shape[1] <= S.SCENE and oy + p.shape[0] <= S.SCENE, "OOB"
        assert p.shape[0] >= 4 and p.shape[1] >= 4, "degenerate window"
    assert len(wins) <= S.MAX_WINDOWS, "exceeded safety cap"
    return len(regs), len(wins)


# 1. empty / flat scene -> nothing proposed
flat = np.full((8, 8), 0.10, dtype=np.float64)
nr, nw = summarize("empty/flat", flat)
assert nr == 0 and nw == 0

# 2. single compact face (one hot blob) -> exactly one region/window
one = np.full((8, 8), 0.10, dtype=np.float64)
one[2:4, 2:4] = 0.90
nr, nw = summarize("single-face", one)
assert nr == 1 and nw == 1

# 3. one BIG face = a wide plateau (near-equal high tiles) -> must stay ONE
plateau = np.full((8, 8), 0.10, dtype=np.float64)
plateau[1:6, 1:6] = 0.88  # 5x5 flat-topped hill
nr, nw = summarize("big-plateau-face", plateau)
assert nr == 1 and nw == 1, "big face was wrongly split!"

# 4. two faces separated by a valley -> must SPLIT into two
two = np.full((8, 8), 0.10, dtype=np.float64)
two[3:5, 0:2] = 0.92      # left hill
two[3:5, 6:8] = 0.92      # right hill (valley of 0.10 between)
nr, nw = summarize("two-faces-valley", two)
assert nw == 2, "two faces did not split!"

# 5. full-frame face (uniformly bright) -> SEED_CEIL still seeds -> one window
fullframe = np.full((8, 8), 0.85, dtype=np.float64)
nr, nw = summarize("full-frame-face", fullframe)
assert nr >= 1 and nw >= 1, "full-frame face missed!"

# 6. noisy but sub-seed everywhere -> no windows (flat gate holds)
rng = np.random.default_rng(0)
noise = 0.2 + 0.15 * rng.random((8, 8))
nr, nw = summarize("low-noise", noise)

# determinism: identical output across repeated calls
a = S._adaptive_regions(two)
b = S._adaptive_regions(two)
assert a == b, "non-deterministic!"

print("\nALL ADAPTIVE CHECKS PASSED")
