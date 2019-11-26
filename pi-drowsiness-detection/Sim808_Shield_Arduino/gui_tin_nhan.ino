void gui_tin_nhan() {
  Serial.println("Gui tin nhan canh bao ");
  at("AT", 500);
  at("AT+CMGF=1", 500);
  at("AT+CSCS=\"GSM\"", 500);
  at("AT+CMGS=\"" + phone_number + "\"", 500);
  at(noi_dung_tin_nhan, 500);
  sim808.write(26);     // ctlrZ
  Serial.println("Da gui tin nhan! Vui long kiem tra");
}
