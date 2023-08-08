from PIL import Image
import cv2
import numpy as np

def enhance_image(image_path):
    # Load the image using PIL
    img_pil = Image.open(image_path).convert("L")  # Convert to grayscale

    if img_pil is None:
        print("Failed to load the image.")
        return

    # Convert PIL image to NumPy array for further processing with cv2
    img_cv2 = np.array(img_pil)

    # Resize the image (reduce size) to 50% of the original
    resized_img_cv2 = cv2.resize(img_cv2, None, fx=0.5, fy=0.5)

    # Increase the brightness of the image (by adding a constant value to all pixels)
    brightened_img = resized_img_cv2 + 100

    # Apply adaptive histogram equalization to enhance contrast using cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(brightened_img)

    # Sharpening using unsharp mask using cv2
    blurred1 = cv2.GaussianBlur(brightened_img, (0, 0), 3)
    sharpened = cv2.addWeighted(brightened_img, 1.5, blurred1, -0.5, 0)

    # Sharpening using unsharp mask using cv2
    blurred2 = cv2.GaussianBlur(equalized_img, (0, 0), 3)
    equalized_sharpened = cv2.addWeighted(equalized_img, 1.5, blurred2, -0.5, 0)
    
    sharpened_equalized = clahe.apply(sharpened)
    # Display the original, resized, brightened, equalized, sharpened, and thresholded images
    cv2.imshow('Original Image', img_cv2)
    cv2.imshow('Brightened Image', brightened_img)
    cv2.imshow('Equalized Image', equalized_img)
    cv2.imshow('Sharpened Image', sharpened)
    cv2.imshow('Equalized_Sharpened Image', equalized_sharpened)
    cv2.imshow('Sharpened_Equalized Image', sharpened_equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\User\Desktop\Github\YOLO(딥러닝)\차량 높이\darktest2.jpg"  # Replace with the path to your image
    enhance_image(image_path)
