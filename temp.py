import time
import cv2
from sdk_t import SyncRemoteInferenceClient

def main():
    # Create the synchronous client. Replace "localhost" with your server's IP if needed.
    print("Creating client")
    client = SyncRemoteInferenceClient("ws://localhost:50004")
    
    client.load_module("obj_detection")
    
    # client.load_module("ocr")
    client.run_module("obj_detection")
    print("Client created")
    
    try:
        while True:
            # Get the latest result for YOLO.
            yolo_result = client.get_current_result("obj_detection")
            print(yolo_result)
            # Get the latest result for Hand Pose.
            # ocr = client.run_module("ocr_detection")
            
            if yolo_result and "image" in yolo_result:
                yolo_img = yolo_result["image"]
                if yolo_img is not None:
                    cv2.imshow("YOLO Output", yolo_img)
                    # cv2.waitKey(0)
            
        #     if ocr and "image" in ocr:
        #         hand_pose_img = ocr["image"]
        #         if hand_pose_img is not None:
        #             cv2.imshow("Hand Pose Output", hand_pose_img)
            
        #     # Press 'q' to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # ocr = client.run_module("ocr")
        # if ocr and "image" in ocr:
        #     hand_pose_img = ocr["image"]
        #     if hand_pose_img is not None:
        #         cv2.imshow("Hand Pose Output", hand_pose_img)
        #         cv2.waitKey(0)
        
        # time.sleep(5)
        
        # ocrr = client.run_module("ocr")
        # if ocrr and "image" in ocrr:
        #     hand_pose_img = ocrr["image"]
        #     if hand_pose_img is not None:
        #         cv2.imshow("Hand Pose Output", hand_pose_img)
        #         cv2.waitKey(0)
        
        # time.sleep(5)
        
    finally:
        client.stop_module("obj_detection")
        # client.stop_module("ocr")
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
