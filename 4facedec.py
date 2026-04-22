import cv2
#Load the pretrained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Load the image
image = cv2.imread(r"C:\CV\download.jpeg")
#Convert the image to Grayscale(face detection works ongraysacle images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Detect faces in the image
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
#Draw Recatangle around the faces
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#Display the result
cv2.imshow('Face Detection',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
