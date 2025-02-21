import cv2
from sdk import SyncRemoteInferenceClient  

def main():
    
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    
    prompt_text = "phone"
    
    result = client.get_result("gdino_inference", prompt=prompt_text)
    if result and "image" in result:
        img = result["image"]
        
        if img is not None and hasattr(img, "shape"):
            cv2.imshow("GDINO Inference", img)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Received an invalid image.")
    else:
        print("No valid result received from gdino inference.")
    
    client.close()

if __name__ == "__main__":
    main()
