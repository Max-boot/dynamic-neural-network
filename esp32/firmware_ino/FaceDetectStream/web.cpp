// ============================================================================
//  web.cpp  -  esp_http_server endpoints.
//
//    GET  /            HTML page: <img> onto the MJPEG stream + upload form.
//    GET  /stream      multipart/x-mixed-replace MJPEG (the overlaid frames).
//    GET  /status      JSON: model loaded?, heap, psram, ip.
//    POST /upload?path=/face_bnn.bin   raw body -> LittleFS (chunked, streamed).
//
//  The stream pulls the newest overlaid JPEG the camera task published into the
//  shared buffer; it never touches the camera directly (single owner rule).
// ============================================================================
#include "web.h"
#include "config.h"
#include "pipeline.h"
#include "model.h"
#include <esp_http_server.h>
#include <esp_heap_caps.h>
#include <LittleFS.h>

static httpd_handle_t s_server = nullptr;
static bool  s_model_loaded = false;
static char  s_ip[24] = "?";

void web_set_status(bool model_loaded, const char* ip) {
  s_model_loaded = model_loaded;
  strncpy(s_ip, ip, sizeof(s_ip) - 1);
  s_ip[sizeof(s_ip) - 1] = 0;
}

// ---- index -----------------------------------------------------------------
static const char INDEX_HTML[] PROGMEM = R"HTML(<!doctype html>
<html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ESP32 Face Cam</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee;text-align:center}
 h1{font-size:1.1rem;padding:.6rem;margin:0;background:#1b1b1b}
 #v{max-width:100%;image-rendering:pixelated;border:1px solid #333;margin-top:.5rem}
 .row{padding:.6rem;font-size:.85rem}
 input[type=file]{color:#eee}
 button{padding:.4rem .8rem;background:#2a6;border:0;color:#fff;border-radius:4px}
 #s{color:#8c8;font-size:.8rem}
</style></head><body>
<h1>ESP32-CAM &mdash; Face Detection Stream</h1>
<img id=v src="/stream">
<div class=row><span id=s>loading status...</span></div>
<div class=row>
 <form id=f>
  <select id=p>
   <option value="/face_bnn.bin">face_bnn.bin</option>
   <option value="/face_saliency.bin">face_saliency.bin</option>
  </select>
  <input type=file id=file>
  <button type=submit>Upload model</button>
 </form>
 <div id=up></div>
</div>
<script>
async function stat(){try{let r=await fetch('/status');let j=await r.json();
 document.getElementById('s').textContent=
 'model:'+(j.model?'loaded':'MISSING')+'  heap:'+(j.heap/1024|0)+'k  psram:'+(j.psram/1024|0)+'k  ip:'+j.ip;
}catch(e){}}
stat();setInterval(stat,3000);
document.getElementById('f').onsubmit=async(e)=>{e.preventDefault();
 let f=document.getElementById('file').files[0];if(!f){return;}
 let p=document.getElementById('p').value;
 document.getElementById('up').textContent='uploading '+f.name+' ('+f.size+' B)...';
 let r=await fetch('/upload?path='+encodeURIComponent(p),{method:'POST',body:f});
 document.getElementById('up').textContent=await r.text();
};
</script>
</body></html>)HTML";

static esp_err_t index_handler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

// ---- MJPEG stream ----------------------------------------------------------
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CT   = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BND  = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t* req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CT);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

  // Local scratch to snapshot the shared JPEG without holding the mutex during send.
  const size_t CAP = 48 * 1024;
  uint8_t* buf = (uint8_t*)heap_caps_malloc(CAP, MALLOC_CAP_SPIRAM);
  if (!buf) buf = (uint8_t*)malloc(CAP);
  if (!buf) return ESP_FAIL;

  char hdr[64];
  size_t last = 0;
  while (true) {
    size_t len = shared_get_jpeg(buf, CAP);
    if (len == 0) { vTaskDelay(pdMS_TO_TICKS(30)); continue; }

    if (httpd_resp_send_chunk(req, STREAM_BND, strlen(STREAM_BND)) != ESP_OK) break;
    int hl = snprintf(hdr, sizeof(hdr), STREAM_PART, (unsigned)len);
    if (httpd_resp_send_chunk(req, hdr, hl) != ESP_OK) break;
    if (httpd_resp_send_chunk(req, (const char*)buf, len) != ESP_OK) break;

    (void)last;
    vTaskDelay(pdMS_TO_TICKS(30));   // ~30 fps ceiling; real rate is camera-bound
  }
  free(buf);
  return ESP_OK;
}

// ---- status ----------------------------------------------------------------
static esp_err_t status_handler(httpd_req_t* req) {
  char js[160];
  int n = snprintf(js, sizeof(js),
    "{\"model\":%s,\"heap\":%u,\"psram\":%u,\"ip\":\"%s\"}",
    s_model_loaded ? "true" : "false",
    (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
    (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
    s_ip);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, js, n);
}

// ---- upload ----------------------------------------------------------------
// Streams the request body straight to a LittleFS file (blobs are ~850 KB, far
// too big to buffer whole). Path comes from ?path=; only whitelisted names OK.
static esp_err_t upload_handler(httpd_req_t* req) {
  char q[64] = {0}, path[48] = {0};
  if (httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK) {
    httpd_query_key_value(q, "path", path, sizeof(path));
  }
  if (strcmp(path, SALIENCY_PATH) != 0 && strcmp(path, BNN_PATH) != 0) {
    httpd_resp_set_status(req, "400 Bad Request");
    return httpd_resp_send(req, "bad path (use /face_bnn.bin or /face_saliency.bin)", HTTPD_RESP_USE_STRLEN);
  }

  File f = LittleFS.open(path, "w");
  if (!f) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    return httpd_resp_send(req, "cannot open file for write", HTTPD_RESP_USE_STRLEN);
  }

  const size_t CH = 4096;
  uint8_t* chunk = (uint8_t*)malloc(CH);
  if (!chunk) { f.close(); return ESP_FAIL; }
  int remaining = req->content_len;
  size_t total = 0;
  bool ok = true;
  while (remaining > 0) {
    int r = httpd_req_recv(req, (char*)chunk, remaining < (int)CH ? remaining : (int)CH);
    if (r <= 0) { ok = false; break; }
    if (f.write(chunk, r) != (size_t)r) { ok = false; break; }
    total += r;
    remaining -= r;
  }
  free(chunk);
  f.close();

  char msg[96];
  if (ok) {
    int n = snprintf(msg, sizeof(msg), "OK wrote %u bytes to %s -- reboot to load", (unsigned)total, path);
    return httpd_resp_send(req, msg, n);
  }
  httpd_resp_set_status(req, "500 Internal Server Error");
  int n = snprintf(msg, sizeof(msg), "write failed after %u bytes", (unsigned)total);
  return httpd_resp_send(req, msg, n);
}

// ---- server bring-up -------------------------------------------------------
bool web_begin() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = HTTP_PORT;
  cfg.ctrl_port   = HTTP_PORT + 1000;
  cfg.max_uri_handlers = 8;
  cfg.lru_purge_enable = true;
  cfg.stack_size = 8192;
  // Pin the HTTP server to core 0 so the inference task keeps core 1.
  cfg.core_id = 0;

  if (httpd_start(&s_server, &cfg) != ESP_OK) {
    Serial.println("[web] httpd_start failed");
    return false;
  }
  httpd_uri_t idx   = { "/",       HTTP_GET,  index_handler,  nullptr };
  httpd_uri_t strm  = { "/stream", HTTP_GET,  stream_handler, nullptr };
  httpd_uri_t stat  = { "/status", HTTP_GET,  status_handler, nullptr };
  httpd_uri_t upl   = { "/upload", HTTP_POST, upload_handler, nullptr };
  httpd_register_uri_handler(s_server, &idx);
  httpd_register_uri_handler(s_server, &strm);
  httpd_register_uri_handler(s_server, &stat);
  httpd_register_uri_handler(s_server, &upl);
  Serial.println("[web] server up");
  return true;
}
