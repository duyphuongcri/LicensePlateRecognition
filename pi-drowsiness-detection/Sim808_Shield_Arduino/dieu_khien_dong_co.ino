void dieu_khien_dong_co(){
  analogWrite(ENA, speed_motor);
  digitalWrite(IN1, 1);
  digitalWrite(IN2, 0);
}
