// ============================================================================
//  camera_pins.h  -  AI-Thinker ESP32-CAM (OV2640) pin map
// ----------------------------------------------------------------------------
//  These are the fixed GPIO assignments for the AI-Thinker board. They match
//  the CAMERA_MODEL_AI_THINKER definition in the esp32-camera examples.
//  NOTE: GPIO 0 is the XCLK *and* the flash-mode strap pin -- keep nothing else
//  on it, and hold IO0->GND only when flashing.
// ============================================================================
#pragma once

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1     // not wired on AI-Thinker
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26     // SCCB SDA
#define SIOC_GPIO_NUM     27     // SCCB SCL

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// On-board lamp / status LED (red LED is GPIO33, active LOW; flash LED GPIO4).
#define LED_GPIO_NUM      33
