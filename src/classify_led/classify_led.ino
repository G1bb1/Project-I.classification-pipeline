/*
File:
classify_led.ino

Description:
Part 5 -- on-device inference on the ESP32, reported on LEDs.
Same classifier as classify_serial.ino: read raw UV and IR, apply the
collection EMA filter, scale with SCALE_BOUNDS, find the grid cell, read
one byte per class from CLASS_TABLE and take the winner -- or UNKNOWN
when even the winner is below REJECT_TAU. One LED per class plus one for
UNKNOWN; exactly one is lit at a time.

Note:
- model.h is written by src/generate_model_header.py; do not edit it.
- Board: ESP32 Dev Module (DevKit V1). Libraries: Adafruit TSL2591,
  Adafruit Unified Sensor. Sensor wiring as in data_collection_UPDATED.ino.
- Works with the Waveshare TSL25911 board (same TSL2591 chip, I2C 0x29):
  VCC -> 3V3, GND -> GND, SDA -> GPIO21, SCL -> GPIO22; INT unused.
- LEDs all sit on one header, the same one as the UV input (GPIO34): the
  five pins right below D34/D35, plus that header's GND pin.
    D32 shade   D33 LED   D25 fluorescent   D26 sun   D27 UNKNOWN
  Each: GPIO -> 220-330 ohm resistor -> LED anode; cathode -> GND.
- If the sensor is missing, all five LEDs blink together.

___________________________________________________
Author      Date                Comment
Gib M.      Oct. 15th, 2026     rev 1
*/
#include <Wire.h>
#include <math.h>
#include <string.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include "model.h"

// --- Sensor set-up: must match data_collection_UPDATED.ino -------------------
// The model was trained on readings taken exactly this way.
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

const int UV_PIN = 34;
const int SDA_PIN = 21;
const int SCL_PIN = 22;

const float ALPHA = 0.2f;                      // EMA smoothing factor
const unsigned long SAMPLE_INTERVAL_MS = 100;  // ~10 Hz, as in collection

float uvFiltered = -1;  // -1 = "not yet initialized"
float irFiltered = -1;
unsigned long lastSampleTime = 0;

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

// Error display: blink every LED forever.
void blinkAllForever() {
  for (bool on = true;; on = !on) {
    for (const ClassLed& led : CLASS_LEDS) digitalWrite(led.pin, on);
    digitalWrite(UNKNOWN_LED_PIN, on);
    delay(250);
  }
}

// --- The classifier (steps 3 to 8) -------------------------------------------
// Returns the class index, or -1 for unknown.
int classify(float uv, float ir) {
  // 3: scale. The bounds are in log units, so log1p comes first.
  float x = (log1pf(uv) - SCALE_BOUNDS[0]) / (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]);
  float y = (log1pf(ir) - SCALE_BOUNDS[2]) / (SCALE_BOUNDS[3] - SCALE_BOUNDS[2]);

  // 4: grid cell. Readings outside the training range clip to the edge.
  int col = constrain((int)(x * GRID_N), 0, GRID_N - 1);
  int row = constrain((int)(y * GRID_N), 0, GRID_N - 1);

  // 5-7: one byte per class, straight from flash; keep the largest.
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

// --- Set-up ------------------------------------------------------------------
void setup() {
  // Match each model class to its LED by name. A name with no LED
  // (model rebuilt with new classes) is an error, shown by blinking.
  bool allFound = true;
  for (int c = 0; c < NUM_CLASSES; c++) {
    ledPin[c] = -1;
    for (const ClassLed& led : CLASS_LEDS) {
      if (strcmp(led.name, CLASS_NAMES[c]) == 0) ledPin[c] = led.pin;
    }
    if (ledPin[c] < 0) allFound = false;
  }
  for (const ClassLed& led : CLASS_LEDS) pinMode(led.pin, OUTPUT);
  pinMode(UNKNOWN_LED_PIN, OUTPUT);
  if (!allFound) blinkAllForever();

  analogReadResolution(12);  // ESP32 ADC: 0-4095
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!tsl.begin()) blinkAllForever();
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);

  showClass(-1);  // nothing known until the first sample
}

// --- Main loop (steps 1, 2 and 9) --------------------------------------------
void loop() {
  // 9: wait for the next sample.
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // 2: raw readings, as in collection.
  int rawUV = analogRead(UV_PIN);
  uint32_t lum = tsl.getFullLuminosity();  // ir<<16 | full
  int rawIR = lum >> 16;                   // IR-only channel

  // Same EMA filter: y_n = alpha*x_n + (1-alpha)*y_(n-1).
  if (uvFiltered < 0) {
    uvFiltered = rawUV;
    irFiltered = rawIR;
  } else {
    uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
    irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
  }

  // 3-8 and report.
  showClass(classify(uvFiltered, irFiltered));
}
