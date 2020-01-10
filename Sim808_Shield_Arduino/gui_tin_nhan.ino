void gui_tin_nhan() {
  //Serial.println("Gui tin nhan ");
  at("AT", 500);
  at("AT+CMGF=1", 500);
  at("AT+CSCS=\"GSM\"", 500);
  at("AT+CMGS=\"" + so_dien_thoai + "\"", 500);
  at(noi_dung_tin_nhan_1, 500);
  sim808.write(26);     // ctlrZ
  delay(5000);
  at("AT", 500);
  at("AT+CMGF=1", 500);
  at("AT+CSCS=\"GSM\"", 500);
  at("AT+CMGS=\"" + so_dien_thoai + "\"", 500);
  at(noi_dung_tin_nhan_2, 500);
  sim808.write(26);     // ctlrZ
  //Serial.println("Da gui tin nhan! Vui long kiem tra");
}
