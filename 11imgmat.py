import cv2 
import numpy as np 
 
def estimate_alpha(image, trimap): 
    # Convert to float 
    image = image.astype(np.float32) / 255.0 
 
    # Normalize trimap to [0, 1] 
    trimap = trimap.astype(np.float32) / 255.0 
 
    # Compute alpha matte using Closed-Form matting 
    foreground = np.where(trimap > 0.95, 1.0, 0.0)  # Foreground mask 
    alpha = np.where(trimap > 0.05, 1.0, 0.0)  # Alpha initialization 
    for _ in range(5):  # Iterative refinement 
        alpha = (image[:, :, 0] - image[:, :, 2] * alpha) / (1e-12 + foreground + (1.0 - trimap) * alpha) 
        alpha = np.clip(alpha, 0, 1)
        return alpha 
# Example usage 
if __name__ == "__main__":
# Read image and trimap 
    image = cv2.imread(r"C:\CV\modelcar.jpeg")
    trimap = cv2.imread(r"C:\CV\modelcar.jpeg",cv2.IMREAD_GRAYSCALE) 
# Estimate alpha matte 
alpha = estimate_alpha(image, trimap) 
# Save or display alpha matte 
cv2.imshow("Alpha Matte", alpha) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
