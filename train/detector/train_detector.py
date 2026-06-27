import os


import json
import torch
import pandas as pd
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

# =====================================================================================
# 1. CUSTOM DATASET CLASS
# =====================================================================================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(os.path.join(root, csv_file))
        # Get a sorted list of unique image filenames
        self.imgs = sorted(self.data['filename'].unique().tolist())

    def __getitem__(self, idx):
        # Load image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "images", img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Convert PIL image to a PyTorch tensor
        img = T.ToTensor()(img)

        # Get bounding box information for this image
        img_annotations = self.data[self.data['filename'] == img_name]
        boxes = []
        for _, row in img_annotations.iterrows():
            shape_attributes = json.loads(row['region_shape_attributes'])
            if shape_attributes['name'] == 'rect':
                x = shape_attributes['x']
                y = shape_attributes['y']
                width = shape_attributes['width']
                height = shape_attributes['height']
                boxes.append([x, y, x + width, y + height])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # There is only one class (plus background), so all labels are 1
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms is not None:
            # Note: torchvision transforms for detection tasks are different
            # They expect a PIL image and the target dictionary
            # A simple ToTensor() is usually sufficient for the model input
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# =====================================================================================
# 2. UTILITY & MODEL HELPER FUNCTIONS
# =====================================================================================
def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    It handles batches of images and targets that may have different sizes.
    """
    return tuple(zip(*batch))

def get_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and modifies the classifier head
    for the specified number of classes.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# =====================================================================================
# 3. TRAINING FUNCTION
# =====================================================================================
def train_model(model, data_loader, optimizer, device, num_epochs):
    print("--- Starting Training ---")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {total_loss/len(data_loader):.4f}")
    
    print("--- Finished Training ---")

# =====================================================================================
# 4. INFERENCE FUNCTION
# =====================================================================================
def run_inference(model, device, image_path, confidence_threshold=0.5):
    print(f"\n--- Running Inference on {os.path.basename(image_path)} ---")
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    # Filter predictions by confidence score
    pred_boxes = prediction[0]['boxes'][prediction[0]['scores'] > confidence_threshold].cpu().numpy()
    pred_scores = prediction[0]['scores'][prediction[0]['scores'] > confidence_threshold].cpu().numpy()
    
    # Create a pandas DataFrame for the tabular output
    results_df = pd.DataFrame({
        'x_min': pred_boxes[:, 0],
        'y_min': pred_boxes[:, 1],
        'x_max': pred_boxes[:, 2],
        'y_max': pred_boxes[:, 3],
        'confidence': pred_scores
    })
    
    return results_df

# =====================================================================================
# 5. MAIN EXECUTION BLOCK
# =====================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    ROOT_DIR = "data"
    CSV_NAME = "via_project_24Aug2025_12h20m_csv.csv"
    MODEL_SAVE_PATH = "fasterrcnn_depl.pth"
    NUM_CLASSES = 2  # 1 class (your object) + 1 background class
    NUM_EPOCHS = 20   # Use more epochs for real training (e.g., 20+)


    # --- (B) Prepare Dataset and DataLoader ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    dataset = CustomDataset(root=ROOT_DIR, csv_file=CSV_NAME)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    # --- (C) Initialize Model and Optimizer ---
    model = get_model(num_classes=NUM_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # --- (D) Train the Model ---
    train_model(model, data_loader, optimizer, device, num_epochs=NUM_EPOCHS)

    # --- (E) Save the Trained Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


