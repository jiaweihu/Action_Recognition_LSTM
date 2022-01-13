import os
import numpy as np
import cv2
from keypoint_func import mediapipe_func


class model_data:
    # Path for exported data, numpy arrays
    data_path = os.path.join('data/mp_data') 
    # Actions that we try to detect
    actions = np.array(['left_hand', 'right_hand', 'both_hand'])
    # Thirty videos worth of data
    no_sequences = 30
    # Videos are going to be 30 frames in length
    sequence_length = 30

    def __init__(self):  
        pass

    def make_data_folder(self):
        for action in self.actions: 
            for sequence in range(self.no_sequences):
                try: 
                    os.makedirs(os.path.join(self.data_path, action, str(sequence)))
                except:
                    pass

    def generateData(self):
        func = mediapipe_func()
        self.make_data_folder()
        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with func.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:            
            # NEW LOOP
            # Loop through actions
            for action in self.actions:
                # Loop through sequences aka videos
                for sequence in range(self.no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):

                        # Read feed
                        ret, frame = cap.read()
                        frame = cv2.flip(frame, 1)

                        # Make detections
                        image, results = func.mediapipe_detection(frame, holistic)
        #                 print(results)

                        # Draw landmarks
                        func.draw_styled_landmarks(image, results)
                        
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION {}'.format(action), (50,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else: 
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = func.extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            break

                    # Break gracefully
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

                # Break gracefully
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break


            cap.release()
            cv2.destroyAllWindows()

    def run(self):
        self.generateData()

if __name__ == '__main__':
    func = model_data()
    func.run()
