import os
import cv2

input_folder = "dataset_resizer/raw_img"
resized_folder = "dataset_resizer/resized_img"
rescaled_folder = "dataset_resizer/rescale_img"
target_size = (256, 256)


os.makedirs(resized_folder, exist_ok=True)
os.makedirs(rescaled_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"File corrup: {filename}")
            continue

        height, width = img.shape[:2]
        base_filename = os.path.splitext(filename)[0]

        if (width, height) == target_size:
            output_path = os.path.join(rescaled_folder, base_filename + ".png")
            cv2.imwrite(output_path, img) 
            print(f" okey format  {base_filename}.png")
            continue

    
        resized_img = cv2.resize(img, target_size)
        output_path = os.path.join(resized_folder, base_filename + ".png")
        cv2.imwrite(output_path, resized_img)
        print(f" Converted: {base_filename}.png")

print("Completed")
