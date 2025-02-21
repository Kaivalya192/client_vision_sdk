import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # Start the Hand Pose Detection module.
    client.start_module("hand_pose_detection")
    
    try:
        while True:
            result = client.get_current_result("hand_pose_detection")
            if result and "image" in result:
                img = result["image"]
                if img is not None:
                    cv2.imshow("Hand Pose Detection", img)
                    # Press 'q' to exit.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        client.stop_module("hand_pose_detection")
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
