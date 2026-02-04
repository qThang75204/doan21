#include <WiFi.h>
#include <esp_camera.h>
#include <WebServer.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===== WIFI =====
const char* ssid = "DUONG KHANG";
const char* password = "THUNGAN*9";

// ===== WEB SERVER =====
WebServer server(80);

// ===== BIẾN NHẬN CHỮ =====
String receivedLetter = "-";

// ===== STREAM =====
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
static const char* STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===== STREAM HANDLER =====
void handle_stream() {
  WiFiClient client = server.client();
  if (!client) return;

  client.println("HTTP/1.1 200 OK");
  client.println("Access-Control-Allow-Origin: *");
  client.printf("Content-Type: %s\r\n\r\n", STREAM_CONTENT_TYPE);

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) continue;

    client.print(STREAM_BOUNDARY);
    client.printf(STREAM_PART, fb->len);
    client.write(fb->buf, fb->len);

    esp_camera_fb_return(fb);
    delay(1);
  }
}

// ===== HTML WEB =====
void handle_root() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ESP32-CAM AI</title>
<style>
body{margin:0;background:black;color:#0f0;font-family:Arial;text-align:center}
#video{width:100vw;max-width:100%;}
#letter{font-size:100px;font-weight:bold;margin-top:10px;}
</style>
</head>
<body>
<img id="video" src="/stream">
<div id="letter">-</div>

<script>
setInterval(()=>{
  fetch('/letter').then(r=>r.text()).then(t=>{
    document.getElementById("letter").innerHTML = t;
  });
},300);
</script>

</body>
</html>
)rawliteral";

  server.send(200, "text/html", html);
}

// ===== API: gửi chữ từ python =====
void handle_set_letter() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    int idx = body.toInt();
    if (idx >= 1 && idx <= 26) {
      receivedLetter = String(char('A' + idx - 1));
      server.send(200, "text/plain", receivedLetter);
      Serial.println("Received: " + receivedLetter);
    } else server.send(400, "text/plain", "Invalid");
  } else server.send(400, "text/plain", "No Data");
}

// ===== API: web đọc chữ =====
void handle_get_letter() {
  server.send(200, "text/plain", receivedLetter);
}

void startCameraServer() {
  server.on("/", HTTP_GET, handle_root);
  server.on("/stream", HTTP_GET, handle_stream);
  server.on("/set_letter", HTTP_POST, handle_set_letter);
  server.on("/letter", HTTP_GET, handle_get_letter);

  server.begin();
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);

  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi OK");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

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

  config.xclk_freq_hz = 24000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_QQVGA;   // 160x120 → rất nhanh → >20fps
  config.jpeg_quality = 10;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera FAIL");
    ESP.restart();
  }

  startCameraServer();

  Serial.println("Camera Ready!");
}

void loop() {
  server.handleClient();
  delay(1); 
}
