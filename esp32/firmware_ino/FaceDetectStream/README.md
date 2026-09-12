# FaceDetectStream — ESP32-CAM face detection over WiFi

Runs the trained face-detection vision cascade **entirely on an AI-Thinker
ESP32-CAM** and livestreams the camera image with detection boxes drawn on it,
over WiFi, with the ESP32 acting as the HTTP server. Inference and the
camera/stream run on **separate cores**.

```
Core 1  inference_task : scene -> Conv+ANFIS saliency -> regions -> refocus
                         windows -> int8 BNN + box head -> detections
Core 0  camera_task    : capture -> build 128x128 scene -> draw boxes -> JPEG
Core 0  http server    : MJPEG stream + model upload + status
```

The two cores exchange only three mutex-guarded objects (the latest scene, the
latest detection list, the latest JPEG) — see `pipeline.cpp`. The camera task is
the sole owner of the camera, so frame buffers are never touched concurrently.

---

## 1. What you need

- **AI-Thinker ESP32-CAM** (OV2640, 4 MB PSRAM) + a USB-serial adapter (FTDI /
  CP2102 at **3.3 V**), or an ESP32-CAM-MB programmer.
- **Arduino IDE** with the **esp32 board package** (Boards Manager → "esp32" by
  Espressif, v2.0.5+ — `fmt2jpg`, LittleFS and `CAMERA_GRAB_LATEST` are all
  present there).
- Python 3 with `requests` for the upload script (`pip install requests`).

### Wiring for flashing
| ESP32-CAM | Programmer |
|-----------|-----------|
| 5V (or 3V3) | 5V / 3V3 |
| GND | GND |
| U0R (RX, GPIO3) | TX |
| U0T (TX, GPIO1) | RX |
| **IO0 → GND** | (only while flashing) |

Hold **IO0 to GND**, press **RST**, upload. Remove the IO0–GND jumper and press
**RST** again to run.

---

## 2. Arduino IDE settings

- **Board:** "AI Thinker ESP32-CAM"
- **Partition Scheme:** "Huge APP (3MB No OTA / 1MB SPIFFS)" — the BNN blob is
  ~855 KB and needs a large app + a filesystem partition.
- **PSRAM:** Enabled
- **Flash Frequency:** 80 MHz, **Flash Mode:** QIO
- **Upload Speed:** 115200 (raise later if stable)

Open `esp32/firmware_ino/FaceDetectStream/FaceDetectStream.ino`. All `.h/.cpp`
in that folder compile as one sketch — keep them together.

---

## 3. WiFi mode (edit `config.h`)

Default is **Access Point** mode — no router needed:

```c
#define WIFI_AP_MODE  1
#define AP_SSID       "ESP32-FaceCam"
#define AP_PASS       "facecam123"     // >= 8 chars, or "" for an open AP
```
Join that WiFi from your phone/PC, then open **http://192.168.4.1/**.

To join your own network instead, set `WIFI_AP_MODE 0` and fill `STA_SSID` /
`STA_PASS`; the device prints its IP on the serial monitor (115200 baud).

---

## 4. Upload the model blobs (one time)

The trained blobs live in `esp32/models/` and are **not** compiled into the
firmware — they are stored on the on-flash filesystem so the sketch fits. After
flashing, the serial log will say the model is missing until you upload them.

**Option A — script (recommended):**
```bash
cd esp32/tools
python upload_model.py                       # AP mode, both blobs -> 192.168.4.1
python upload_model.py --host 192.168.1.42   # STA mode: use the printed IP
```

**Option B — web UI:** open the device page, pick a blob name, choose the file
from `esp32/models/`, click *Upload*. Do both files.

Then **reboot the ESP32** (press RST or power-cycle). It parses the blobs into
PSRAM on boot; the status line turns to `model:loaded`.

---

## 5. View the stream

Open **http://192.168.4.1/** (AP) or **http://<printed-ip>/** (STA). You get the
live grayscale image at 256×256 with green boxes on detected faces. The status
line shows model state, free heap/PSRAM and IP.

Endpoints: `/` (page), `/stream` (MJPEG), `/status` (JSON), `/upload` (POST).

---

## 6. Calibrating detection (important)

The class-index mapping in `config.h` is an **assumption** until you confirm it:

```c
#define FACE_CLASS  0     // which BNN logit means "face"
#define BG_CLASS    1
#define GATE        0.6f  // min class prob to keep a window
#define BG_REJECT_P 0.7f  // drop windows confidently classed as background
#define REGION_THR  0.5f  // saliency tile active threshold
```

Confirm and tune these on the host **before** trusting the device, using the
NumPy reference that the firmware mirrors 1:1:

```bash
cd esp32/tools
python sim_pipeline.py --image face.jpg    --mode refocus
python sim_pipeline.py --image notface.jpg --mode refocus
```

It prints the saliency map stats, the region list, and per-window class probs +
decoded box. Whichever class index is high on faces and low otherwise is
`FACE_CLASS`. Copy the thresholds you validated into `config.h` and re-flash.

The firmware runs `--mode refocus` (one square crop per salient region, side
clamped to `[WIN_MIN, WIN_MAX]`, centered on the saliency centroid). This is the
strategy the training pipeline's `hybrid2` uses for normal-sized regions; it is
chosen over full `hybrid2` because each BNN pass is heavy on a plain ESP32 (no
SIMD) and `MAX_WINDOWS` bounds the per-frame compute. `--mode bbox` (raw region
box) is also available (`REGION_MODE` in `config.h`).

---

## 7. Files

| File | Role |
|------|------|
| `FaceDetectStream.ino` | setup, dual-core task creation, camera init, scene build, box overlay, JPEG encode |
| `config.h` | all tunables: WiFi, paths/sizes, thresholds, region mode |
| `camera_pins.h` | AI-Thinker GPIO map |
| `model.h/.cpp` | LittleFS mount, blob upload store, PSRAM parse into typed pointers |
| `nn.h/.cpp` | forward pass: saliency (Conv+ANFIS), regions (BFS), windows, int8 BNN + box |
| `pipeline.h/.cpp` | mutex-guarded shared state + the core-1 inference loop |
| `web.h/.cpp` | esp_http_server: index, MJPEG stream, upload, status |
| `../tools/sim_pipeline.py` | NumPy reference the firmware mirrors — use it to calibrate |
| `../tools/upload_model.py` | push blobs to the device over WiFi |
| `../models/*.bin` | the trained blobs (`face_saliency.bin` 3608 B, `face_bnn.bin` 875928 B) |

---

## 8. Notes & limits

- **Precision:** the host reference accumulates convs in float64; the ESP32 has
  no hardware double, so `nn.cpp` accumulates in float32. This stays inside the
  verified BNN tolerance (argmax stable, <0.15 prob drift) and does not change
  region/box results in practice.
- **Frame rate** is inference-bound. The stream shows the newest camera frame at
  ~15–30 fps; detection boxes update as fast as the cascade finishes (may lag
  the image by a frame or two). Lower `MAX_WINDOWS` or `WIN_MAX` for speed.
- The `esp32/blobio/verify_*.py` scripts compare the blobs against the original
  PyTorch checkpoints. Those `.pt` files are **not** in this repo, so only the
  NumPy-vs-blob checks run; `sim_pipeline.py` needs no `.pt` and is the
  executable spec used here.
