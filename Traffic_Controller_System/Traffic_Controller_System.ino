 
#define latch1    4
#define clock1    3
#define data1     5
#define latch2    10
#define clock2    9
#define data2     11
#define do1       7
#define vang1     6 
#define xanh1     2
#define do2       13
#define xanh2     8
#define vang2     12
int  xanhdem, dodem, dem, chuc1,chuc2, donvi1, donvi2;
int giaydung=980;

const unsigned char MA7DOAN[16]= {0XC0,0XF9,0xA4,0XB0,0X99,0X92,0X82,0XF8,0X80,0X90,0x88, 0x83, 0xC6, 0xA1, 0x86, 0x8E};
void setup()
{
  pinMode(latch1, OUTPUT);
  pinMode(do1, OUTPUT);
  pinMode(xanh1, OUTPUT);
  pinMode(vang1, OUTPUT);
  pinMode(latch2, OUTPUT);
  pinMode(do2, OUTPUT);
  pinMode(xanh2, OUTPUT);
  pinMode(vang2, OUTPUT);
  Serial.begin(9600);
  delay(3000);
  }

void loop()
{
  time_set(20);
  /*
//  time_set(20);
  //time_set(40);
  //time_set(30);
  //time_set(30);
  */
}
void time_set(int thoigian)
{
  for(int j=thoigian; j>=0; j--)
  {
    if(j>=4)
    {
      xanhdem=(j-4);
      digitalWrite(xanh1, 1);
      digitalWrite(do2, 1);
      digitalWrite(do1, 0);
      digitalWrite(vang2, 0);
      Serial.println(1); // Switch red green light to red light
    }
    dodem=j;
    if(j<4)
    {
      xanhdem=j;
      digitalWrite(xanh1, 0);
      digitalWrite(vang1, 1);
     
      
    }
    giaima(xanhdem, dodem);
    delay(giaydung);
  }
  for(int j=thoigian; j>=0; j--)
  {
    if(j>=4)
    {
      xanhdem=(j-4);
      digitalWrite(vang1, 0);
      digitalWrite(do1, 1);
      digitalWrite(do2, 0);
      digitalWrite(xanh2, 1);
       Serial.println(0); // Switch green, yellow light to red light
    }
    dodem=j;
    if(j<4)
    {
      xanhdem=j;
      digitalWrite(xanh2, 0);
      digitalWrite(vang2, 1);
    }
    giaima(dodem, xanhdem);
    delay(giaydung);
  }
}
void giaima(int so1,int so2)
{
  digitalWrite(latch1, 0);
  digitalWrite(latch2, 0);
  chuc1 = MA7DOAN[so1/10];
  if (chuc1== 0XC0) chuc1 = 0xff;
  chuc2 = MA7DOAN[so2/10] ;
  if (chuc2== 0XC0) chuc2 = 0xff;
  donvi1 = MA7DOAN[so1%10] ;
  donvi2 = MA7DOAN[so2%10] ;
  shiftOut(data1, clock1, chuc1);   
  shiftOut(data1, clock1, donvi1);
  shiftOut(data2, clock2, chuc2);   
  shiftOut(data2, clock2, donvi2);
  digitalWrite(latch1, 1);
  digitalWrite(latch2, 1);
}
void shiftOut(int myDataPin, int myClockPin, byte myDataOut) {
  int i=0;
  int pinState;
  pinMode(myClockPin, OUTPUT);
  pinMode(myDataPin, OUTPUT);
  digitalWrite(myDataPin, 0);
  digitalWrite(myClockPin, 0);
  for (i=7; i>=0; i--)  {
  digitalWrite(myClockPin, 0);
  if ( myDataOut & (1<<i) )
    pinState= 1;
  else  pinState= 0;
  digitalWrite(myDataPin, pinState);
  digitalWrite(myClockPin, 1);
  digitalWrite(myDataPin, 0);
  }
  digitalWrite(myClockPin, 0);
}
