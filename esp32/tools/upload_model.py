#!/usr/bin/env python3
# ============================================================================
#  upload_model.py  -  Push the trained blobs to the ESP32-CAM over WiFi.
#
#  The firmware serves POST /upload?path=<dest> and streams the raw body into
#  a LittleFS file. After both blobs are up, the device must reboot once to
#  parse them (the script triggers nothing destructive; just power-cycle or
#  press RST, or the firmware re-parses on next boot).
#
#  Usage:
#     # AP mode (default firmware): join WiFi "ESP32-FaceCam" first, then:
#     python upload_model.py                      # uploads both blobs to 192.168.4.1
#     python upload_model.py --host 192.168.1.42  # STA mode: use the printed IP
#     python upload_model.py --only bnn           # just one blob
#
#  Requires: pip install requests
# ============================================================================
import argparse
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("This script needs 'requests'.  Install with:  pip install requests")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.normpath(os.path.join(_HERE, "..", "models"))

# (local file, device path, expected size) -- sizes are the firmware's sanity check.
BLOBS = {
    "saliency": ("face_saliency.bin", "/face_saliency.bin", 3608),
    "bnn":      ("face_bnn.bin",      "/face_bnn.bin",      875928),
}


def upload_one(host, port, local, dest, expect):
    path = os.path.join(_MODELS, local)
    if not os.path.isfile(path):
        sys.exit(f"missing blob: {path}")
    size = os.path.getsize(path)
    if size != expect:
        print(f"  WARNING: {local} is {size} B, firmware expects {expect} B "
              f"(it will reject a mismatched size on load).")
    url = f"http://{host}:{port}/upload"
    with open(path, "rb") as fh:
        data = fh.read()
    print(f"  POST {local} ({size} B) -> {dest} ...", end="", flush=True)
    t0 = time.time()
    r = requests.post(url, params={"path": dest}, data=data,
                      headers={"Content-Type": "application/octet-stream"},
                      timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        print(f" FAILED ({r.status_code}): {r.text}")
        return False
    print(f" ok  [{dt:.1f}s]  {r.text}")
    return True


def status(host, port):
    try:
        r = requests.get(f"http://{host}:{port}/status", timeout=5)
        print(f"  device status: {r.text}")
    except Exception as e:
        print(f"  (status unavailable: {e})")


def main():
    ap = argparse.ArgumentParser(description="Upload face-detection blobs to the ESP32-CAM.")
    ap.add_argument("--host", default="192.168.4.1", help="device IP (AP mode default 192.168.4.1)")
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--only", choices=list(BLOBS.keys()),
                    help="upload just this blob (default: both)")
    args = ap.parse_args()

    keys = [args.only] if args.only else ["saliency", "bnn"]
    print(f"Target: http://{args.host}:{args.port}/")
    status(args.host, args.port)

    ok = True
    for k in keys:
        local, dest, expect = BLOBS[k]
        ok &= upload_one(args.host, args.port, local, dest, expect)

    if ok:
        print("\nDone. Reboot the ESP32 (press RST or power-cycle) so it parses the blobs.")
        print("Then open the stream at  http://%s:%d/" % (args.host, args.port))
    else:
        print("\nOne or more uploads failed -- see messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
