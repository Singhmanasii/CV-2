import cv2
from matplotlib import pyplot as plt

image_path = "C:/Users/singh/Downloads/object.jpeg"
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image {image_path}")
else:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cascade_path = 'C:/Users/singh/Downloads/haarcascade_frontalcatface.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Failed to load cascade classifier {cascade_path}")
        detections = []
    else:
        detections = cascade.detectMultiScale(image_gray, minSize=(30, 30))
        detection_count = len(detections)
        print(f"Number of detections: {detection_count}")

    for (x, y, width, height) in detections:
        cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
