/*
File:
classify.ino

Description:
Part 4 -- on-device inference on the ESP32, reported on serial AND LEDs.
Every sample: read raw UV and IR, apply the same EMA filter used during
collection, take log1p, scale with SCALE_BOUNDS, find the grid cell, read
one byte per class from CLASS_TABLE and take the winner -- or UNKNOWN when
even the winner is below REJECT_TAU. The result is printed on serial and
shown on one LED per class (plus one for UNKNOWN); exactly one is lit.

It also measures what the report has to compare against the predicted
footprint: at start-up it prints the model's size and where it sits in
memory, the free heap and stack, and the average time per query; every
serial line then carries that query's own time.

Validation: send L:<label> (the same command as data_collection_UPDATED.ino)
while holding the sensor under a known light, to record a fresh held-out
batch as CSV rows, each with the predicted class next to the true label.

Note:
- model.h is written by src/generate_model_header.py; do not edit it.
  It must sit in the same folder as this file: classify/classify.ino and
  classify/model.h.
- Board: ESP32 Dev Module (DevKit V1). Libraries: Adafruit TSL2591,
  Adafruit Unified Sensor. Sensor wiring as in data_collection_UPDATED.ino.
- Works with the Waveshare TSL25911 board: same TSL2591 chip, I2C address
  0x29 and chip ID 0x50, which is all the Adafruit library checks.
  Waveshare VCC -> 3V3, GND -> GND, SDA -> GPIO21, SCL -> GPIO22; INT unused.
- UV: GUVA-S12SD OUT -> GPIO34 (ADC1, so it keeps working with WiFi on).
- LEDs, each GPIO -> 220-330 ohm resistor -> LED anode; cathode -> GND:
    D32 shade   D33 LED   D25 fluorescent   D26 sun   D27 UNKNOWN
- Serial Monitor at 115200 baud.
    untagged line:  UV, IR -> class  (time)
    L:<label> rows: uv,ir,predicted,label,us   (send L: to stop)
    R toggles rows: raw_uv,raw_ir,filt_uv,filt_ir,predicted
- Errors (sensor missing, or a model class with no LED): message on
  serial, then all five LEDs blink together.

___________________________________________________
Author      Date                Comment
Gib M.      Oct. 15th, 2026     rev 1 -- classify_serial.ino and
                                classify_led.ino, separate sketches.
Gib M.      Oct. 16th, 2026     rev 2 -- merged into one sketch reporting
                                on serial and LEDs; saturation check uses
                                the 100 ms count ceiling; R command streams
                                raw vs filtered readings.
*/
#include <Wire.h>
#include <math.h>
#include <string.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include "model.h"

// --- Sensor set-up: must match data_collection_UPDATED.ino -------------------
// The model was trained on readings taken exactly this way. A different
// gain, integration time, filter constant or sample rate shifts the
// readings, and the grid cells no longer mean the same thing.
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

const int UV_PIN = 34;
const int SDA_PIN = 21;
const int SCL_PIN = 22;

const float ALPHA = 0.2f;                      // EMA smoothing factor
const unsigned long SAMPLE_INTERVAL_MS = 100;  // ~10 Hz, as in collection

// At 100 ms integration each TSL2591 channel tops out at 36863 counts, not
// 65535, so that is where saturation shows up.
const uint16_t TSL_MAX_COUNT = 36863;

float uvFiltered = -1;  // -1 = "not yet initialized"
float irFiltered = -1;
unsigned long lastSampleTime = 0;

String trueLabel = "";  // empty = not recording a held-out batch
bool rawMode = false;   // R toggles: stream raw and filtered side by side

// --- LED display -------------------------------------------------------------
// GPIO 32, 33, 25, 26, 27 are consecutive on the DevKit's header and are
// all safe outputs (none is a boot strapping pin). Ordered dimmest to
// brightest light, so the lit LED moves along the row with the light.
struct ClassLed {
  const char* name;
  int pin;
};
const ClassLed CLASS_LEDS[] = {
  {"shade", 32}, {"LED", 33}, {"fluorescent", 25}, {"sun", 26},
};
const int UNKNOWN_LED_PIN = 27;

// LED pin for each class index, looked up by name: the cluster numbers in
// model.h can change when the model is rebuilt, the names should not.
int ledPin[NUM_CLASSES];

void showClass(int c) {
  for (int k = 0; k < NUM_CLASSES; k++) {
    digitalWrite(ledPin[k], k == c ? HIGH : LOW);
  }
  digitalWrite(UNKNOWN_LED_PIN, c < 0 ? HIGH : LOW);
}

// Error display: print why, then blink every LED forever.
void failForever(const char* message) {
  Serial.print("# ERROR: ");
  Serial.println(message);
  for (bool on = true;; on = !on) {
    for (const ClassLed& led : CLASS_LEDS) digitalWrite(led.pin, on);
    digitalWrite(UNKNOWN_LED_PIN, on);
    delay(250);
  }
}

// --- The classifier (steps 3 to 8) -------------------------------------------
// Returns the class index, or -1 for unknown.
int classify(float uv, float ir) {
  // 3: scale. The bounds are in log units, so log1p comes first, exactly
  //    as transform_features does in Python.
  float x = (log1pf(uv) - SCALE_BOUNDS[0]) / (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]);
  float y = (log1pf(ir) - SCALE_BOUNDS[2]) / (SCALE_BOUNDS[3] - SCALE_BOUNDS[2]);

  // 4: grid cell. Readings outside the training range clip to the edge.
  int col = constrain((int)(x * GRID_N), 0, GRID_N - 1);
  int row = constrain((int)(y * GRID_N), 0, GRID_N - 1);

  // 5-7: one byte per class, straight from flash; keep the largest.
  // Strict > keeps the first class on a tie, like numpy's argmax.
  int best = 0;
  float bestValue = -1.0f;
  for (int c = 0; c < NUM_CLASSES; c++) {
    float v = pgm_read_byte(&CLASS_TABLE[c][row][col]) / 255.0f;
    if (v > bestValue) {
      bestValue = v;
      best = c;
    }
  }

  // 8: too weak to trust -> unknown.
  return (bestValue < REJECT_TAU) ? -1 : best;
}

const char* nameOf(int c) {
  return (c < 0) ? "UNKNOWN" : CLASS_NAMES[c];
}

// Cycle counter, not micros(): one query takes only a few microseconds,
// below micros()'s 1 us resolution.
float cyclesToUs(uint32_t cycles) {
  return cycles / (float)getCpuFrequencyMhz();
}

// --- Footprint and timing report ---------------------------------------------
void reportFootprint() {
  const unsigned modelBytes = sizeof(CLASS_TABLE) + sizeof(SCALE_BOUNDS);
  Serial.printf("# model: CLASS_TABLE %u B + SCALE_BOUNDS %u B = %u B "
                "(predicted %u = %u*%u^2 + 16)\n",
                (unsigned)sizeof(CLASS_TABLE), (unsigned)sizeof(SCALE_BOUNDS),
                modelBytes, NUM_CLASSES * GRID_N * GRID_N + 16u,
                NUM_CLASSES, GRID_N);

  // On the original ESP32, 0x3F400000-0x3F7FFFFF is flash mapped as data
  // (DROM): a table there costs flash but no SRAM.
  uint32_t addr = (uint32_t)CLASS_TABLE;
  bool inFlash = addr >= 0x3F400000 && addr < 0x3F800000;
  Serial.printf("# CLASS_TABLE at 0x%08X -> %s\n", (unsigned)addr,
                inFlash ? "flash (DROM), no SRAM used" : "NOT in flash");

  Serial.printf("# sketch %u B of %u B flash chip\n",
                (unsigned)ESP.getSketchSize(), (unsigned)ESP.getFlashChipSize());
  Serial.printf("# heap: %u B free of %u B; loop stack: %u B never used\n",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getHeapSize(),
                (unsigned)uxTaskGetStackHighWaterMark(NULL));
}

// Time many queries spread over the whole input range, so every grid
// cell's path is exercised, and average them.
void benchmark() {
  const int N = 10000;
  volatile int sink = 0;  // stops the compiler from dropping the calls
  uint32_t t0 = ESP.getCycleCount();
  for (int i = 0; i < N; i++) {
    float uv = (i % 100) * 40.0f;            // 0 .. 3960
    float ir = 100.0f + (i / 100) * 320.0f;  // 100 .. 31780
    sink += classify(uv, ir);
  }
  uint32_t cycles = ESP.getCycleCount() - t0;
  Serial.printf("# time per query: %.2f us average over %d queries "
                "(%u cycles each, CPU %u MHz)\n",
                cyclesToUs(cycles) / N, N, (unsigned)(cycles / N),
                (unsigned)getCpuFrequencyMhz());
}

// --- Set-up ------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  // LEDs first, so an error can be shown on them.
  for (const ClassLed& led : CLASS_LEDS) pinMode(led.pin, OUTPUT);
  pinMode(UNKNOWN_LED_PIN, OUTPUT);

  // Match each model class to its LED by name. A name with no LED
  // (model rebuilt with new classes) is an error.
  for (int c = 0; c < NUM_CLASSES; c++) {
    ledPin[c] = -1;
    for (const ClassLed& led : CLASS_LEDS) {
      if (strcmp(led.name, CLASS_NAMES[c]) == 0) ledPin[c] = led.pin;
    }
    if (ledPin[c] < 0) failForever("a class in model.h has no LED; update CLASS_LEDS.");
  }

  analogReadResolution(12);  // ESP32 ADC: 0-4095
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!tsl.begin()) {
    failForever("TSL2591 not found. Check wiring (SDA=21, SCL=22) and power.");
  }
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);

  benchmark();
  reportFootprint();
  Serial.println("# Ready. Send L:<label> to record a held-out batch, L: to stop;");
  Serial.println("# send R to toggle raw-vs-filtered streaming.");

  showClass(-1);  // nothing known until the first sample
}

// --- Main loop (steps 1, 2 and 9) --------------------------------------------
void loop() {
  handleSerialCommands();

  // 9: wait for the next sample.
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // 2: raw readings, as in collection: UV from the ADC, IR from the
  //    TSL2591's IR-only channel.
  int rawUV = analogRead(UV_PIN);
  uint32_t lum = tsl.getFullLuminosity();  // ir<<16 | full
  uint16_t full = lum & 0xFFFF;
  uint16_t ir = lum >> 16;
  if (full >= TSL_MAX_COUNT || ir >= TSL_MAX_COUNT) {
    Serial.println("# WARNING: TSL2591 saturated.");
  }
  if (rawUV >= 4095) {
    Serial.println("# WARNING: UV ADC at its 4095 ceiling.");
  }
  int rawIR = ir;

  // Same EMA filter: y_n = alpha*x_n + (1-alpha)*y_(n-1).
  if (uvFiltered < 0) {
    uvFiltered = rawUV;
    irFiltered = rawIR;
  } else {
    uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
    irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
  }

  // 3-8, timed on their own: the sensor read above takes ~100 ms and
  // would swamp the classifier's few microseconds.
  uint32_t t0 = ESP.getCycleCount();
  int c = classify(uvFiltered, irFiltered);
  float us = cyclesToUs(ESP.getCycleCount() - t0);

  // Report on both outputs.
  showClass(c);
  if (rawMode) {
    // Filter evidence: CSV row raw_uv,raw_ir,filt_uv,filt_ir,predicted
    Serial.printf("%d,%d,%.2f,%.2f,%s\n", rawUV, rawIR, uvFiltered,
                  irFiltered, nameOf(c));
  } else if (trueLabel.length() > 0) {
    // CSV row for the held-out batch: uv,ir,predicted,label,us
    Serial.printf("%.2f,%.2f,%s,%s,%.2f\n", uvFiltered, irFiltered,
                  nameOf(c), trueLabel.c_str(), us);
  } else {
    Serial.printf("UV %7.2f  IR %8.2f  ->  %-11s (%.2f us)\n",
                  uvFiltered, irFiltered, nameOf(c), us);
  }
}

// L:<label> starts recording a held-out batch under that true label;
// L: (empty) stops. Same commands as the collection sketch.
void handleSerialCommands() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();

  // R: toggle raw-vs-filtered streaming (the filter-evidence capture).
  if (line == "R") {
    rawMode = !rawMode;
    Serial.println(rawMode ? "# raw_uv,raw_ir,filt_uv,filt_ir,predicted"
                           : "# Raw streaming off.");
    return;
  }
  if (!line.startsWith("L:")) return;

  String newLabel = line.substring(2);
  newLabel.trim();
  if (newLabel == trueLabel) return;
  trueLabel = newLabel;
  if (trueLabel.length() > 0) {
    Serial.print("# Recording held-out batch for label: ");
    Serial.println(trueLabel);
  } else {
    Serial.println("# Recording paused.");
  }
}