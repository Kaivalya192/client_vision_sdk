import cv2
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:8765")
    
    # Define a prompt for the LLava inference module.
    prompt_text = "Describe the scene."
    
    result = client.get_result("llava_inference", prompt=prompt_text)
    
    if result:
        # Access and print the result text.
        if "result" in result:
            print("Result text:", result["result"])
        
        # If LLava returns an image result:
        if "image" in result:
            img = result["image"]
            if img is not None:
                cv2.imshow("LLava Inference", img)
                print("Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Received an invalid image.")
    else:
        print("No valid result received from LLava inference.")
    
    client.close()

if __name__ == "__main__":
    main()
