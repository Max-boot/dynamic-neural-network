// ============================================================================
//  web.h  -  HTTP server: index page, MJPEG stream, model upload, status.
// ============================================================================
#pragma once
#include <Arduino.h>

// Start the esp_http_server on HTTP_PORT. Call after WiFi is up and shared
// buffers exist. Returns true on success.
bool web_begin();

// Set by the .ino so /status can report model + core state.
void web_set_status(bool model_loaded, const char* ip);
