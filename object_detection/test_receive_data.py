import serial

try:
    arduino_lighttraffic = serial.Serial("COM3", 9600 ,timeout=1)
    arduino_moduleSim = serial.Serial("COM5", 9600 ,timeout=1)
    print("Found out Arduino Uno device")
except:
    print("Please checl the port")

while True:
    data = arduino_lighttraffic.readline()
    data = data.decode("utf-8").rstrip('\r\n') 
    # if data == "":
    #     print("none")
    if data == "0":
        print("do")
        arduino_moduleSim.write(b"0989522890")
    if data == "1":
        arduino_moduleSim.write(b"11")
        print("xanh")

    
