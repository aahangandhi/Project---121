import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Setting frame width and frame height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

# Loading the mountain image
mountain = cv2.imread('mount_everest.jpg')

# Resizing the mountain image as 640 x 480
resized_mountain = cv2.resize(mountain, (640, 480))

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip it
        frame = cv2.flip(frame, 1)

        # Converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating thresholds
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        # Thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # Inverting the mask
        inverted_mask = cv2.bitwise_not(mask)

        # Bitwise and operation to extract foreground/person
        foreground = cv2.bitwise_and(frame, frame, mask=inverted_mask)

        # Bitwise and operation to replace white background with mountain image
        background = cv2.bitwise_and(resized_mountain, resized_mountain, mask=mask)

        # Final image
        final_image = cv2.add(foreground, background)

        # Show it
        cv2.imshow('frame', final_image)

        # Wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# Release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
