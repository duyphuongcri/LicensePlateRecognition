#include "SoftwareSerial.h"
SoftwareSerial sim808(8, 7);
String inputString;
String thoi_gian = "";
String noi_dung_tin_nhan = "Canh bao: Tai xe dieu khien voi xe bien so 51F-9999 dang trong tinh trang buon ngu vao luc: ";
void goi_dien();
void gui_tin_nhan();
void gui_tin_nhan_on();
void gui_tin_nhan_off();

//Khai bao chan dong co
int Relay_PI_Ard = 4;
int Switch       = 5;
int interrupt_E_button     = 2;
int interrupt_Reset_button = 3;
int Red_Led      = 6;
int ENA          = 9;
int IN1          = 10;
int IN2          = 11;
int Relay_NC_Ard = 12;
int Buzzer       = 13;
int speed_motor = 175; // nam trong khoang 0-255
int ON = 0;
int sta_E = 0;
int sta_Reset_time = 0;

//String phone_number = "0394108627";
String phone_number = "0327405097";
//String phone_number = "0911715102";

void setup() {
  pinMode(Relay_PI_Ard, OUTPUT);
  pinMode(Switch, INPUT);
  pinMode(interrupt_E_button, INPUT_PULLUP);
  pinMode(interrupt_Reset_button, INPUT_PULLUP);
  pinMode(Red_Led, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(Relay_NC_Ard, OUTPUT);
  pinMode(Buzzer, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(interrupt_E_button), check_button_buzzer, FALLING);
  digitalWrite(Buzzer, 1);
  Serial.begin(115200);
  sim808.begin(9600);
  if ((digitalRead(Switch) == 1) && (ON == 0))
  {
    gui_tin_nhan_on();
    digitalWrite(Relay_PI_Ard, 0);
    delay(1500);
    digitalWrite(Relay_NC_Ard, 1);
    ON = 1;
    Serial.println("Da BAT thiet bi");
  }

  // Test call
  //  goi_dien();
  //Test sms
  //gui_tin_nhan();
}

void loop() {
  if ((digitalRead(Switch) == 0) && (ON == 1))
  {
    gui_tin_nhan_off();
    delay(3000);
    Serial.println("Da TAT thiet bi");
    ON = 0;
    delay(2000);
    digitalWrite(Relay_PI_Ard, 1);
    digitalWrite(Relay_NC_Ard, 0);
  }

  while (Serial.available())
  {
    inputString = Serial.readString();
    //    inputString = Serial.readStringUntil(' ');
    inputString.toLowerCase();
    Serial.print("Tin hieu nhan:   ");
    Serial.println(inputString);
    thoi_gian = inputString.substring(8);
    Serial.print("Thoi gian:   ");
    Serial.println(thoi_gian);
    noi_dung_tin_nhan += thoi_gian;
    Serial.println(noi_dung_tin_nhan);
    //    thoi_gian= Serial.parseInt();
    if (inputString.startsWith("warning"))
    {
      goi_dien();
      delay(2000);
      gui_tin_nhan();
      digitalWrite(Red_Led, 1);
      dieu_khien_dong_co();
      delay(7000);
      dung_dong_co();
      digitalWrite(Red_Led, 0);

    }
    //--------------------------------------------------
    if (inputString.startsWith("3h"))
    {
      goi_dien();
      delay(2000);
      digitalWrite(Red_Led, 1);
      gui_tin_nhan_3h();
      dieu_khien_dong_co();
      delay(7000);
      dung_dong_co();
      //digitalWrite(13, 1);
      digitalWrite(Red_Led, 0);

    }
    // ------------------------------------------------ -
    if (inputString.startsWith("call"))
    {
      //      dieu_khien_dong_co();
      goi_dien();
      delay(15000);
      dung_dong_co();
      digitalWrite(13, 1);
    }

    // ------------------------------------------------ -
    if (inputString.startsWith("mesg_warning"))
    {
      gui_tin_nhan();
    }

    // ------------------------------------------------ -
    if (inputString.startsWith("mesg_on"))
    {
      gui_tin_nhan_on();
    }

    // ------------------------------------------------ -
    if (inputString.startsWith("mesg_off"))
    {
      gui_tin_nhan_off();
    }

    // ------------------------------------------------ -
    if (inputString.startsWith("motor"))
    {
      dieu_khien_dong_co();
      Serial.print("Dang dieu khien dong co");
    }
  }
}
void at(String _atcm, unsigned long _dl) {
  sim808.print(_atcm + "\r\n");
  delay(_dl);
}
