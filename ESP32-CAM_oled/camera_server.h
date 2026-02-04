#pragma once
#include <WebServer.h>
#include <esp_camera.h>

WebServer server(80);

static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
static const char* STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

void handle_jpg_stream(void) {
  WiFiClient client = server.client();
  if (!client) return;

  client.println("HTTP/1.1 200 OK");
  client.println("Access-Control-Allow-Origin: *");
  client.printf("Content-Type: %s\r\n\r\n", STREAM_CONTENT_TYPE);

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;

    client.print(STREAM_BOUNDARY);
    client.printf(STREAM_PART, fb->len);
    client.write(fb->buf, fb->len);

    esp_camera_fb_return(fb);

    yield();   // cực kỳ quan trọng
  }
}

void handle_jpg(void) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handle_root(void) {
  server.send(200, "text/html",
    "<html><body style='margin:0;background:black;'>"
    "<img src='/stream' style='width:100%;height:auto;'>"
    "</body></html>");
}

void startCameraServer() {
  server.on("/", HTTP_GET, handle_root);
  server.on("/jpg", HTTP_GET, handle_jpg);
  server.on("/stream", HTTP_GET, handle_jpg_stream);

  server.begin();
  Serial.println("HTTP server started");
}
