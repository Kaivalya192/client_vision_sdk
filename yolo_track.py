import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # Start the YOLO Segmentation & Tracking module.
    client.start_module("yolo_seg_track")
    
    try:
        while True:
            result = client.get_current_result("yolo_seg_track")
            if result and "image" in result:
                img = result["image"]
                if img is not None:
                    cv2.imshow("YOLO Segmentation & Tracking", img)
                    # Press 'q' to exit.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        client.stop_module("yolo_seg_track")
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
