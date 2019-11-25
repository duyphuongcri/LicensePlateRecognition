void gui_tin_nhan_3h() {
  Serial.println("Gui tin nhan canh bao ");
  at("AT", 500);
  at("AT+CMGF=1", 500);
  at("AT+CSCS=\"GSM\"", 500);
  at("AT+CMGS=\"" + phone_number + "\"", 500);
  at("Tai xe tren xe 51F-9999 da lai xe lien tuc 3h45ph luc " + thoi_gian , 500);
  sim808.write(26);     // ctlrZ
  Serial.println("Da gui tin nhan! Vui long kiem tra");
}
