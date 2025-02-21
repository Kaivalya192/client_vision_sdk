import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # Define a prompt for the OWL-ViT module.
    prompt_text = "book"
    
    result = client.get_result("owl_vit", prompt=prompt_text)
    if result and "image" in result:
        img = result["image"]
        if img is not None:
            cv2.imshow("OWL-ViT Inference", img)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Received an invalid image.")
    else:
        print("No valid result received from OWL-ViT inference.")
    
    client.close()

if __name__ == "__main__":
    main()
