from ultralytics import YOLO
import torch
import cv2
import os
from PIL import Image
import numpy as np
import sys
# from CNN_based.inference import load_model
# from CNN_based.train import SRCNN
import torch.nn as nn
from iterative_backprojection import IBP
from Self_Similarity.res.python.self_similarity import super_res_self_sim

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def run_yolo(source, file_name, task):

    # GPU support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO("yolov8n.pt")
    model.to(device)

    # Run inference
    results = model(source, device=device)

    # Vehicle class IDs in COCO
    vehicle_ids = {2, 3, 4, 5, 6, 7, 8}

    output_string = ""

    avg_confidence = 0.0
    total_vehicles = 0

    for result in results:
        final_boxes = []
        boxes = result.boxes


        for box in boxes:
            cls = int(box.cls[0])

            # Only keep vehicles
            if cls not in vehicle_ids:
                continue
            else:
                final_boxes.append(box)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            avg_confidence += conf
            total_vehicles += 1

            cls_name = model.names[cls]

            output_string += (
                f"Class: {cls_name}, "
                f"Conf: {conf:.3f}, "
                f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n"
            )

        result.boxes = final_boxes
        avg_confidence = (avg_confidence / total_vehicles) if total_vehicles > 0 else 0.0

    print("Vehicle Detections:\n", output_string if output_string else "No vehicles detected.")

    # Save annotated output image
    result_img = results[0].plot()
    annotated_dir = f"annotated_outputs/{task}"
    os.makedirs(annotated_dir, exist_ok=True)
    out_path = f"{annotated_dir}/{file_name}"
    cv2.imwrite(out_path, result_img)
    # print("Annotated image saved to:", out_path)

    return avg_confidence, out_path


if __name__ == "__main__":
    car_images_dir = "D:\\DIP Project\\car_testing"
    factor = 3
    # tasks = ["Bicubic", "SRCNN", "IBP"]
    tasks = ["self_local"]

    model = SRCNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load("D:\\DIP Project\\srcnn_x3.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    for task in tasks:
        task_dir = os.path.join(car_images_dir, task)
        os.makedirs(task_dir, exist_ok=True)

    for img_name in os.listdir(car_images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        img_path = os.path.join(car_images_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        print(f"Processing image: {img_name}")
        for task in tasks:
            print(f"  Applying task: {task}")
            
            
            task_img_path = os.path.join(car_images_dir, task, img_name)
            if task == "Bicubic":
                img_pil = Image.fromarray(img)
                new_size = (img_pil.width//factor, img_pil.height//factor)
                img_resized = img_pil.resize(new_size, Image.BICUBIC)
                new_size = (img_resized.width*factor, img_resized.height*factor)
                img_resized = img_resized.resize(new_size, Image.BICUBIC)
                img_resized.save(task_img_path)
            elif task == "SRCNN":
                img_pil = Image.fromarray(img)
                new_size = (img_pil.width//factor, img_pil.height//factor)
                img_resized = img_pil.resize(new_size, Image.BICUBIC)
                new_size = (img_resized.width*factor, img_resized.height*factor)
                img_resized = img_resized.resize(new_size, Image.BICUBIC)

                img_resized = np.array(img_resized).astype(np.float32) / 255.0
                input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                output_img = output_tensor.squeeze().cpu().numpy()
                output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(output_img).save(task_img_path)
            elif task == "IBP":

                img_pil = Image.fromarray(img)
                new_size = (img_pil.width//factor, img_pil.height//factor)
                img_resized = img_pil.resize(new_size, Image.BICUBIC)
                img_resized = np.array(img_resized).astype(np.float32)
                ibp_img = IBP(img_resized, factor, factor)
                ibp_img = np.clip(ibp_img, 0, 255).astype(np.uint8)
                Image.fromarray(ibp_img).save(task_img_path)

            elif task == "self_local":
                img_pil = Image.fromarray(img)
                new_size = (img_pil.width//factor, img_pil.height//factor)
                img_resized = img_pil.resize(new_size, Image.BICUBIC)
                img_resized = np.array(img_resized).astype(np.float32)/255.0
                sr_image = super_res_self_sim(img_resized, s=factor)
                sr_image = Image.fromarray(np.astype(sr_image*255.0, np.uint8)).convert("L")
                sr_image.save(task_img_path)

    # tasks = ["Bicubic", "SRCNN", "IBP"]

    # for task in tasks:
    #     img_dir = f"D:\\DIP Project\\car_testing\\{task}"
    #     os.makedirs(f"annotated_outputs/{task}", exist_ok=True)
    #     output_dir = f"annotated_outputs/{task}"

    #     average_confidences = []
    #     for img_name in os.listdir(img_dir):
    #         if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    #             continue
    #         print(f"Running YOLO on {task} image: {img_name}")
            
    #         img_path = os.path.join(img_dir, img_name)
    #         avg_conf, out_path = run_yolo(img_path, img_name, task)

    #         average_confidences.append(avg_conf)

    #         print(f"Average confidence for {img_name} ({task}): {avg_conf:.3f}\n")

    #     overall_avg_conf = (sum(average_confidences) / len(average_confidences)) if average_confidences else 0.0
    #     print(f"Overall average confidence for {task}: {overall_avg_conf:.3f}\n")

    #     txt_file = open(f"{output_dir}/average_confidence.txt", "w")
    #     txt_file.write(f"Overall average confidence for {task}: {overall_avg_conf:.3f}\n")
    #     txt_file.close()

