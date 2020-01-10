import serial
import time
try:
    arduino_lighttraffic = serial.Serial("COM9", 9600 ,timeout=1)
    arduino_moduleSim = serial.Serial("COM6", 115200 ,timeout=1)
    print("Found out Arduino Uno device")
except:
    print("Please checl the port")
# a = 1
# while True:
#     message = "warning:038362659119:46:50 27/11/201971B400472"
#     #arduino_lighttraffic.write(message.encode())
#     #time.sleep(0.1)
#     data = arduino_lighttraffic.readline()
#     data = data.decode("utf-8").rstrip('\r\n') 
#     print(data)
    # if data == "":
    #     print("none")
    # if data == "0":
    #     print("do")
    #     #arduino_moduleSim.write(b"0989522890")
    # if data == "1":
    #     #arduino_moduleSim.write(b"11")
    #     print("xanh")

    
