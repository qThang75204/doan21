#include <WiFi.h>
#include <esp_camera.h>
#include <WebServer.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===== WIFI - THAY ĐỔI THEO WIFI CỦA BẠN =====
const char* ssid = "DUONG KHANG";
const char* password = "THUNGAN*9";

// ===== WEB SERVER =====
WebServer server(80);

// ===== STREAM LIMITER =====
static int activeStreams = 0;
const int MAX_STREAMS = 2;

// ===== STREAM VIDEO =====
void handle_stream() {
  if (activeStreams >= MAX_STREAMS) {
    server.send(503, "text/plain", "Stream busy");
    return;
  }
  
  WiFiClient client = server.client();
  if (!client) return;

  activeStreams++;
  Serial.println(" Stream connected");

  client.println("HTTP/1.1 200 OK");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Content-Type: multipart/x-mixed-replace;boundary=frame\r\n");

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      delay(100);
      continue;
    }

    client.print("\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ");
    client.print(fb->len);
    client.print("\r\n\r\n");
    
    if (client.write(fb->buf, fb->len) != fb->len) {
      esp_camera_fb_return(fb);
      break;
    }
    
    esp_camera_fb_return(fb);
    delay(30);
  }
  
  activeStreams--;
  Serial.println(" Stream disconnected");
}

// ===== HTML PAGE =====
void handle_root() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ESP32-CAM Stream</title>
<style>
body{margin:0;padding:20px;background:#1a1a1a;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;font-family:Arial}
.container{background:#2d2d2d;border-radius:15px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,0.5);max-width:800px;width:100%}
h1{color:#0f0;text-align:center;margin:0 0 20px 0;font-size:24px}
.info{color:#888;text-align:center;margin-bottom:15px;font-size:14px}
.url{color:#0f0;background:#000;padding:10px;border-radius:5px;font-family:monospace;margin:10px 0}
#video{width:100%;border-radius:10px;background:#000}
</style>
</head>
<body>
<div class='container'>
  <h1>📹 ESP32-CAM Live Stream</h1>
  <div class='info'>Sử dụng URL này trong Python code:</div>
  <div class='url'>http://)" + WiFi.localIP().toString() + R"(/stream</div>
  <img id='video' src='/stream'>
</div>
</body>
</html>
)";
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("\n====================================");
  Serial.println("  ESP32-CAM VIDEO STREAM ONLY");
  Serial.println("====================================");

  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  // WIFI
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(300);
    Serial.print(".");
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n WiFi Connected!");
    Serial.print(" IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.println("\n STREAM URL (Copy vào Python code):");
    Serial.print("   http://");
    Serial.print(WiFi.localIP());
    Serial.println("/stream");
    Serial.println("\n Web Browser:");
    Serial.print("   http://");
    Serial.println(WiFi.localIP());
    Serial.println("====================================\n");
  } else {
    Serial.println("\n WiFi FAILED!");
  }

  // ===== CAMERA CONFIG =====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_DRAM;

  esp_err_t err = esp_camera_init(&config);
  
  if (err != ESP_OK) {
    Serial.printf(" Camera init failed: 0x%x\n", err);
    Serial.println(" Trying lower resolution...");
    
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    err = esp_camera_init(&config);
    
    if (err != ESP_OK) {
      Serial.println(" Failed! Restarting...");
      delay(3000);
      ESP.restart();
    }
  }

  // Camera settings
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_lenc(s, 1);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
  }

  server.on("/", HTTP_GET, handle_root);
  server.on("/stream", HTTP_GET, handle_stream);
  server.begin();

  Serial.println(" Camera ready!");
  Serial.printf(" Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.println("====================================");
  Serial.println(" READY - Waiting for connections...");
  Serial.println("====================================\n");
}

void loop() {
  server.handleClient();
  yield();
}
