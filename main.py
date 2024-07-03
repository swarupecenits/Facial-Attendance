import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "images"
images = []
classNames = []

# List all subfolders in the parent directory
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

print("Total students detected:", len(subfolders))

for subfolder in subfolders:
    personName = os.path.basename(subfolder)
    # List all files in the current subfolder
    imageFiles = os.listdir(subfolder)
    for imageFile in imageFiles:
        curImg = cv2.imread(os.path.join(subfolder, imageFile))
        if curImg is not None:
            images.append(curImg)
            classNames.append(personName)

# print(classNames) #students names


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    file_exists = os.path.isfile("Attendance.csv")

    with open('Attendence.csv', 'r+') as f:
        if not file_exists:
            f.writelines("Name,Date,Time\n")

        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime("%Y-%m-%d")
            timeString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dateString},{timeString}")


encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if (
            matches[matchIndex] and faceDis[matchIndex] < 0.6
        ):  # You can adjust the threshold value
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        cv2.putText(
            img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2
        )

        if name != "UNKNOWN":
            markAttendance(name)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
