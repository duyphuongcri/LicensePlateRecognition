void gui_tin_nhan_off() {
  Serial.println("Gui tin nhan TAT thiet bi");
  at("AT", 500);
  at("AT+CMGF=1", 500);
  at("AT+CSCS=\"GSM\"", 500);
  at("AT+CMGS=\"" + phone_number + "\"", 500);
  at("Thiet bi canh bao buon ngu tren xe 51F-9999 da bi TAT", 500);
  sim808.write(26);     // ctlrZ
  Serial.println("Da gui tin nhan! Vui long kiem tra");
}
