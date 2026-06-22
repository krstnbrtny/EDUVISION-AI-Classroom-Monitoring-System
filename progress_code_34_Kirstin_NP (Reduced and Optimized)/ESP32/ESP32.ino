#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include "driver/i2s.h"

// ===============================
// WIFI CONFIG
// ===============================
const char* ssid = "CHRISTIAN HOME";
const char* password = "Philippians413";
const char* serverUrl = "http://192.168.0.104:5000/api/noise_level";

// ===============================
// RELAY / LED
// ===============================
const int RELAY_PIN = 2;
const float NOISE_THRESHOLD = 50.0;
unsigned long relayActivatedTime = 0; // timestamp when relay was turned on
const unsigned long COOLDOWN_MS = 500; // 0.5 seconds

// ===============================
// I2S MICROPHONE PINS (INMP441)
// ===============================
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 26

#define I2S_PORT I2S_NUM_0

void setup() {

  Serial.begin(115200);

  // LED / Relay
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  // ===============================
  // WIFI CONNECTION
  // ===============================
  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi Connected!");

  // ===============================
  // I2S CONFIGURATION
  // ===============================

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);

  Serial.println("I2S Microphone Initialized");
}

void loop() {

  int32_t samples[1024];
  size_t bytesRead;

  // ===============================
  // READ MICROPHONE DATA
  // ===============================

  i2s_read(I2S_PORT, samples, sizeof(samples), &bytesRead, portMAX_DELAY);

  int sampleCount = bytesRead / sizeof(int32_t);

  float sum = 0;

  for (int i = 0; i < sampleCount; i++) {

    float normalized = samples[i] / 2147483648.0;

    sum += normalized * normalized;
  }

  float rms = sqrt(sum / sampleCount);

  float db = 20 * log10(rms + 1e-6) + 90;

  Serial.printf("Noise Level: %.2f dB\n", db);

  // ===============================
  // NOISE THRESHOLD + COOLDOWN
  // ===============================
  if (db >= NOISE_THRESHOLD) {
    // Turn relay ON only if it's OFF
    if (digitalRead(RELAY_PIN) == LOW) {
      digitalWrite(RELAY_PIN, HIGH);
      relayActivatedTime = millis(); // start cooldown timer
      Serial.println("⚠ Noise threshold exceeded! LED ON");
    }
  }

  // Check if cooldown expired
  if (digitalRead(RELAY_PIN) == HIGH && (millis() - relayActivatedTime >= COOLDOWN_MS)) {
    digitalWrite(RELAY_PIN, LOW);
    Serial.println("Cooldown complete. LED OFF");
  }

  // ===============================
  // SEND DATA TO FLASK SERVER
  // ===============================

  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;

    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"noise_level\": " + String(db) + "}";

    int response = http.POST(payload);

    Serial.print("Server response: ");
    Serial.println(response);

    http.end();
  }

  delay(1000);
}