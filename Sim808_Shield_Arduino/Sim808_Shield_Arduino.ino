#include "SoftwareSerial.h"
SoftwareSerial sim808(6, 8); // rx-tx
String data;
String inputString;
String thoi_gian = "";
String bien_so = "";
String noi_dung_tin_nhan_1;
String noi_dung_tin_nhan_2;
void goi_dien();
void gui_tin_nhan();


String so_dien_thoai;
//String so_dien_thoai = "0911715102";

void setup() {
  Serial.begin(115200);
  sim808.begin(9600);

  // Test call
  //  goi_dien();
  //Test sm
  //gui_tin_nhan();
}

void loop() {
  
  while (Serial.available())
  {  
    inputString = Serial.readString();
    //inputString = Serial.readStringUntil(' ');
    //inputString.toLowerCase();
    //Serial.print("Tin hieu nhan:   ");
    //Serial.println(inputString);

    so_dien_thoai = inputString.substring(8, 18);
    //Serial.print("So dien thoai:   ");
    //Serial.println(so_dien_thoai);
    
    thoi_gian = inputString.substring(18, 37);
    //Serial.print("Thoi gian:   ");
    //Serial.println(thoi_gian); 

    bien_so = inputString.substring(37);
    //Serial.print("Bien so:   ");
    //Serial.println(bien_so); 
          
    //thoi_gian= Serial.parseInt();
    if (inputString.startsWith("redlight"))
    {
      noi_dung_tin_nhan_1 = "[Thong bao]\n Anh (Chi) su dung phuong tien mang bien so " + bien_so + " da vuot den do, luc " + thoi_gian + " tai giao lo Phu Hung.";
      noi_dung_tin_nhan_2 = "Vui long truy cap http://vuotdendotpbentre.ddns.net de tra cuu thong tin chi tiet va phuong thuc nop phat. \n -Tran trong-";
      gui_tin_nhan();
    }
    else if (inputString.startsWith("hailoi")){
      noi_dung_tin_nhan_1 = "[Thong bao]\n Anh (Chi) su dung phuong tien mang bien so " + bien_so + " khong doi non bao hiem va vuot den do, luc " + thoi_gian + " tai giao lo Phu Hung.";
      noi_dung_tin_nhan_2 = "Vui long truy cap http://vuotdendotpbentre.ddns.net de tra cuu thong tin chi tiet va phuong thuc nop phat. \n -Tran trong-";
      gui_tin_nhan();
    }
    else if(inputString.startsWith("nohelmet")){
      noi_dung_tin_nhan_1 = "[Thong bao]\n Anh (Chi) su dung phuong tien mang bien so " + bien_so + " khong doi non bao hiem, luc " + thoi_gian + " tai giao lo Phu Hung.";
      noi_dung_tin_nhan_2 = "Vui long truy cap http://vuotdendotpbentre.ddns.net de tra cuu thong tin chi tiet va phuong thuc nop phat. \n -Tran trong-";
      gui_tin_nhan();
    }  
  }
}
void at(String _atcm, unsigned long _dl) {
  sim808.print(_atcm + "\r\n");
  delay(_dl);
}
