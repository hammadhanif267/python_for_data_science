# import numpy as np
# import cv2 as cv

# # Read webcam using OpenCV
# cap = cv.VideoCapture(0)  # Change 1 to 0 for default webcam

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
# else:
#     ret, frame = cap.read()
#     while ret:
#         ret, frame = cap.read()
#         cv.imshow("Webcam", frame)

#         # Press 'q' to exit
#         if cv.waitKey(25) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv.destroyAllWindows()


# for gray-scale


# import numpy as np
# import cv2 as cv

# # Read webcam using OpenCV
# cap = cv.VideoCapture(0)  # Use 0 for the default webcam

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
# else:
#     ret, frame = cap.read()
#     while ret:
#         ret, frame = cap.read()

#         # Convert frame to grayscale
#         gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#         # Display grayscale webcam feed
#         cv.imshow("Grayscale Webcam", gray_frame)

#         # Press 'q' to exit
#         if cv.waitKey(25) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv.destroyAllWindows()


# for colored-scale

import cv2 as cv

# Read webcam using OpenCV
cap = cv.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        
        # Display colored webcam feed
        cv.imshow("Color Webcam", frame)

        # Press 'q' to exit
        if cv.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

