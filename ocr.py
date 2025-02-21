import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # For OCR detection, no prompt is needed by default.
    result = client.get_result("ocr_detection")
    
    if result:
        # If the result includes an image with OCR annotations:
        if "image" in result:
            img = result["image"]
            if img is not None:
                cv2.imshow("OCR Detection", img)
                print("Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Received an invalid image.")
        else:
            # Alternatively, if the result returns text, print it.
            print("OCR Results:", result.get("text", "No text available"))
    else:
        print("No valid result received from OCR detection.")
    
    client.close()

if __name__ == "__main__":
    main()
