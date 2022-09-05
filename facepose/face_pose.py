import cv2
import matplotlib.pyplot as plt
from rich.console import Console

# read from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Console().print(
    f"camera is open 📸  {cap.isOpened()} width : \
{cap.get(3)} height : {cap.get(4)}"
)
# get a frame
ret, frame = cap.read()
if frame is not None:
    Console().print(
        f"frame dtype :arrow_right: {frame.dtype}, max \
         :arrow_right: {frame.max()}, min. :arrow_right: {frame.min()}"
    )
    cv2.imwrite("frame.jpg", frame)

f, ax = plt.subplots(figsize=(10, 10))
frame = plt.imread("frame.jpg")
ax.axis("off")
plt.imshow(frame)
plt.show()
