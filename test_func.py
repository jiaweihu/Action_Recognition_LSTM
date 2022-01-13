from common import mediapipe_func
import cv2

def displayHolistic():
    common_func = mediapipe_func()
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with common_func.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = common_func.mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            common_func.draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

