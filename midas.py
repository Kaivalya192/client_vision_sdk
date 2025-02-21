import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # Start the MiDaS Depth Estimation module.
    client.start_module("midas_depth")
    
    try:
        while True:
            result = client.get_current_result("midas_depth")
            if result and "image" in result:
                img = result["image"]
                if img is not None:
                    cv2.imshow("MiDaS Depth Estimation", img)
                    # Press 'q' to exit.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        client.stop_module("midas_depth")
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
