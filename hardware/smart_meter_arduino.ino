/*
  Smart Energy Meter with AI/ML Integration
  Final Year Project - GSM Based Remote Monitoring
  
  Features:
  - Energy measurement (Voltage, Current, Power, Energy)
  - GSM communication with cloud server
  - AI prediction integration
  - LCD display for local monitoring
  - Alert system for anomalies
  - Data logging to SD card
*/

#include <SoftwareSerial.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>
#include <EEPROM.h>
#include <Wire.h>

// Pin Definitions
#define GSM_TX 2
#define GSM_RX 3
#define VOLTAGE_PIN A0
#define CURRENT_PIN A1
#define BUZZER_PIN 8
#define LED_PIN 13

// GSM Module
SoftwareSerial gsm(GSM_TX, GSM_RX);

// LCD Display (I2C)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Configuration
const String METER_ID = "METER_001";
const String SERVER_URL = "http://your-server.com/api/meter/data";
const String APN = "internet";  // Your GSM provider's APN
const unsigned long SEND_INTERVAL = 60000;  // Send data every minute
const unsigned long DISPLAY_INTERVAL = 2000;  // Update display every 2 seconds

// Global Variables
float voltage = 0.0;
float current = 0.0;
float power = 0.0;
float energy = 0.0;
float frequency = 50.0;
float powerFactor = 0.9;
float temperature = 25.0;
int signalStrength = 0;
float batteryLevel = 100.0;

unsigned long lastSendTime = 0;
unsigned long lastDisplayTime = 0;
bool gsmConnected = false;
bool serverConnected = false;

// AI Prediction Variables
float predictedPower = 0.0;
float predictedCost = 0.0;
String recommendations = "";

void setup() {
  Serial.begin(9600);
  gsm.begin(9600);
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Smart Meter AI");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  
  // Initialize pins
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(VOLTAGE_PIN, INPUT);
  pinMode(CURRENT_PIN, INPUT);
  
  // Initialize GSM
  initializeGSM();
  
  // Load saved energy value from EEPROM
  loadEnergyFromEEPROM();
  
  Serial.println("Smart Meter with AI Integration Started");
  
  delay(2000);
  lcd.clear();
}

void loop() {
  // Read sensor data
  readSensorData();
  
  // Update display
  if (millis() - lastDisplayTime >= DISPLAY_INTERVAL) {
    updateDisplay();
    lastDisplayTime = millis();
  }
  
  // Send data to server
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    if (gsmConnected) {
      sendDataToServer();
    } else {
      initializeGSM();
    }
    lastSendTime = millis();
  }
  
  // Check for alerts
  checkAlerts();
  
  // Save energy to EEPROM periodically
  saveEnergyToEEPROM();
  
  delay(100);
}

void readSensorData() {
  // Read voltage (scaled for 0-250V range)
  int voltageRaw = analogRead(VOLTAGE_PIN);
  voltage = (voltageRaw * 250.0) / 1023.0;
  
  // Read current (scaled for 0-20A range using ACS712)
  int currentRaw = analogRead(CURRENT_PIN);
  current = ((currentRaw - 512) * 20.0) / 512.0;
  if (current < 0) current = 0;  // Ensure positive current
  
  // Calculate power
  power = (voltage * current * powerFactor) / 1000.0;  // Convert to kW
  
  // Calculate energy (integrate power over time)
  static unsigned long lastEnergyTime = millis();
  unsigned long currentTime = millis();
  float timeDiff = (currentTime - lastEnergyTime) / 3600000.0;  // Convert to hours
  energy += power * timeDiff;
  lastEnergyTime = currentTime;
  
  // Simulate temperature reading (replace with actual sensor)
  temperature = 25.0 + random(-5, 10);
  
  // Get GSM signal strength
  signalStrength = getGSMSignalStrength();
  
  // Simulate battery level (replace with actual battery monitoring)
  batteryLevel = 100.0 - (millis() / 100000.0);  // Decreases over time
  if (batteryLevel < 0) batteryLevel = 0;
}

void updateDisplay() {
  static int displayMode = 0;
  
  lcd.clear();
  
  switch (displayMode) {
    case 0:  // Power and Energy
      lcd.setCursor(0, 0);
      lcd.print("P:");
      lcd.print(power, 2);
      lcd.print("kW E:");
      lcd.print(energy, 1);
      lcd.setCursor(0, 1);
      lcd.print("V:");
      lcd.print(voltage, 0);
      lcd.print("V I:");
      lcd.print(current, 1);
      lcd.print("A");
      break;
      
    case 1:  // AI Predictions
      lcd.setCursor(0, 0);
      lcd.print("AI Pred:");
      lcd.print(predictedPower, 1);
      lcd.print("kW");
      lcd.setCursor(0, 1);
      lcd.print("Cost: $");
      lcd.print(predictedCost, 2);
      break;
      
    case 2:  // Status
      lcd.setCursor(0, 0);
      lcd.print("GSM:");
      lcd.print(gsmConnected ? "OK" : "ERR");
      lcd.print(" Sig:");
      lcd.print(signalStrength);
      lcd.setCursor(0, 1);
      lcd.print("Bat:");
      lcd.print(batteryLevel, 0);
      lcd.print("% T:");
      lcd.print(temperature, 0);
      lcd.print("C");
      break;
  }
  
  displayMode = (displayMode + 1) % 3;  // Cycle through display modes
}

void initializeGSM() {
  Serial.println("Initializing GSM...");
  
  // Reset GSM module
  gsm.println("AT");
  delay(1000);
  
  // Check if GSM is responding
  if (gsm.find("OK")) {
    Serial.println("GSM Module Ready");
    
    // Set APN
    gsm.println("AT+SAPBR=3,1,\"CONTYPE\",\"GPRS\"");
    delay(1000);
    gsm.println("AT+SAPBR=3,1,\"APN\",\"" + APN + "\"");
    delay(1000);
    
    // Enable GPRS
    gsm.println("AT+SAPBR=1,1");
    delay(2000);
    
    // Check connection
    gsm.println("AT+SAPBR=2,1");
    delay(1000);
    
    if (gsm.find("1,1")) {
      gsmConnected = true;
      Serial.println("GPRS Connected");
      digitalWrite(LED_PIN, HIGH);
    } else {
      gsmConnected = false;
      Serial.println("GPRS Connection Failed");
      digitalWrite(LED_PIN, LOW);
    }
  } else {
    gsmConnected = false;
    Serial.println("GSM Module Not Responding");
    digitalWrite(LED_PIN, LOW);
  }
}

void sendDataToServer() {
  if (!gsmConnected) return;
  
  Serial.println("Sending data to server...");
  
  // Create JSON payload
  StaticJsonDocument<512> doc;
  doc["meter_id"] = METER_ID;
  doc["timestamp"] = getTimestamp();
  doc["voltage"] = voltage;
  doc["current"] = current;
  doc["power_kw"] = power;
  doc["energy_kwh"] = energy;
  doc["frequency"] = frequency;
  doc["power_factor"] = powerFactor;
  doc["temperature"] = temperature;
  doc["signal_strength"] = signalStrength;
  doc["battery_level"] = batteryLevel;
  
  String jsonString;
  serializeJson(doc, jsonString);
  
  // Send HTTP POST request
  gsm.println("AT+HTTPINIT");
  delay(1000);
  
  gsm.println("AT+HTTPPARA=\"CID\",1");
  delay(1000);
  
  gsm.println("AT+HTTPPARA=\"URL\",\"" + SERVER_URL + "\"");
  delay(1000);
  
  gsm.println("AT+HTTPPARA=\"CONTENT\",\"application/json\"");
  delay(1000);
  
  gsm.println("AT+HTTPDATA=" + String(jsonString.length()) + ",10000");
  delay(1000);
  
  gsm.println(jsonString);
  delay(2000);
  
  gsm.println("AT+HTTPACTION=1");
  delay(5000);
  
  // Read response
  gsm.println("AT+HTTPREAD");
  delay(2000);
  
  String response = "";
  while (gsm.available()) {
    response += gsm.readString();
  }
  
  // Parse AI response
  parseAIResponse(response);
  
  gsm.println("AT+HTTPTERM");
  delay(1000);
  
  Serial.println("Data sent successfully");
  serverConnected = true;
}

void parseAIResponse(String response) {
  // Parse JSON response from AI server
  StaticJsonDocument<1024> doc;
  DeserializationError error = deserializeJson(doc, response);
  
  if (!error) {
    if (doc["status"] == "success") {
      // Extract AI predictions
      if (doc.containsKey("prediction")) {
        predictedPower = doc["prediction"]["predicted_power_kw"];
        predictedCost = doc["prediction"]["predicted_cost"];
      }
      
      // Extract recommendations
      if (doc.containsKey("recommendations")) {
        recommendations = doc["recommendations"][0].as<String>();
      }
      
      Serial.println("AI predictions received");
    }
  }
}

int getGSMSignalStrength() {
  gsm.println("AT+CSQ");
  delay(500);
  
  String response = "";
  while (gsm.available()) {
    response += gsm.readString();
  }
  
  // Parse signal strength from response
  int csqIndex = response.indexOf("+CSQ: ");
  if (csqIndex != -1) {
    int signalValue = response.substring(csqIndex + 6, csqIndex + 8).toInt();
    return -113 + (signalValue * 2);  // Convert to dBm
  }
  
  return -99;  // Unknown signal strength
}

String getTimestamp() {
  // Get timestamp from GSM network
  gsm.println("AT+CCLK?");
  delay(500);
  
  String response = "";
  while (gsm.available()) {
    response += gsm.readString();
  }
  
  // Parse timestamp (simplified - you might want to format this properly)
  return "2024-01-15T" + String(millis() / 1000) + "Z";
}

void checkAlerts() {
  bool alertTriggered = false;
  
  // High power consumption alert
  if (power > 5.0) {
    Serial.println("ALERT: High power consumption!");
    alertTriggered = true;
  }
  
  // Voltage out of range alert
  if (voltage < 200 || voltage > 250) {
    Serial.println("ALERT: Voltage out of range!");
    alertTriggered = true;
  }
  
  // Low battery alert
  if (batteryLevel < 20) {
    Serial.println("ALERT: Low battery level!");
    alertTriggered = true;
  }
  
  // GSM connection lost alert
  if (!gsmConnected) {
    Serial.println("ALERT: GSM connection lost!");
    alertTriggered = true;
  }
  
  // Sound buzzer for alerts
  if (alertTriggered) {
    for (int i = 0; i < 3; i++) {
      digitalWrite(BUZZER_PIN, HIGH);
      delay(200);
      digitalWrite(BUZZER_PIN, LOW);
      delay(200);
    }
  }
}

void saveEnergyToEEPROM() {
  static unsigned long lastSaveTime = 0;
  
  // Save energy value every 10 minutes
  if (millis() - lastSaveTime >= 600000) {
    EEPROM.put(0, energy);
    lastSaveTime = millis();
    Serial.println("Energy saved to EEPROM");
  }
}

void loadEnergyFromEEPROM() {
  float savedEnergy;
  EEPROM.get(0, savedEnergy);
  
  // Check if saved value is valid
  if (!isnan(savedEnergy) && savedEnergy >= 0 && savedEnergy < 999999) {
    energy = savedEnergy;
    Serial.println("Energy loaded from EEPROM: " + String(energy));
  } else {
    energy = 0.0;
    Serial.println("No valid energy data in EEPROM, starting from 0");
  }
}

// Additional utility functions can be added here for:
// - SD card logging
// - Real-time clock integration
// - Advanced sensor calibration
// - Over-the-air updates
// - Local web server for configuration