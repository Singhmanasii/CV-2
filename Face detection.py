import cv2
import matplotlib.pyplot as plt

# Path to the image
imgPath = 'C:/Users/singh/Downloads/people smiling.jpeg'

# Load the image
img = cv2.imread(imgPath)

# Check if the image is loaded
if img is None:
    print("Error loading image. Check the file path.")
else:
    print("Image loaded successfully.")
    print("Image shape:", img.shape)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces with different parameters
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces detected.")
    else:
        print("Faces detected:", len(faces))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
