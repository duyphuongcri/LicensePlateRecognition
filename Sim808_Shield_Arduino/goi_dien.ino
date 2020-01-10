void goi_dien(){
  Serial.println("Calling...");
  at("AT",1000);
  at("ATD"+so_dien_thoai+";",10000);
  at("ATH",1000);
  Serial.println("Called");
}
