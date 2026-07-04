import torch
import pandas as pd
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from tqdm import tqdm
import os


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

def run_inference(model, device, image_path, confidence_threshold=0.5):
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

def visualize_predictions(image_path, results_df, output_path="prediction.png"):
    """
    Draws bounding boxes and confidence scores on an image and saves it.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, row in results_df.iterrows():
        # Get coordinates
        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        confidence = row['confidence']
        
        # Draw the bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline="red", 
            width=3
        )
        
        # Prepare the text
        text = f"{confidence:.2f}"
        
        # Draw a small background rectangle for the text
        text_bbox = draw.textbbox((x_min, y_min - 12), text) # Pillow >= 10.0.0
        draw.rectangle(text_bbox, fill="red")
        
        # Draw the confidence score text
        draw.text(
            (x_min, y_min - 12), 
            text, 
            fill="white"
        )

    # Save the image
    img.save(output_path)
    print(f"Saved visualization to {output_path}")

    # Optionally, display the image
    #img.show()
    
def predict_bb(proj_dir,model= "model/fasterrcnn_depl.pth"):
    
    print('Detecting objects...')
    # --- (B) Prepare Dataset and DataLoader ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # --- (F) Run Inference ---
    # Re-initialize the model for inference (a good practice)
    inference_model = get_model(num_classes=2)    
    inference_model.load_state_dict(torch.load(model, map_location=torch.device(device)))
    
    # Get and print tabular results
    res=pd.DataFrame()
    INPUT_FOLDER=proj_dir+'/input_raw'
    for im in tqdm(os.listdir(INPUT_FOLDER)):
        TEST_IMAGE_PATH=INPUT_FOLDER+'/'+im
        inference_results = run_inference(inference_model, device, TEST_IMAGE_PATH,
                                          confidence_threshold=0.8)
        inference_results['filename']=[im]*len(inference_results)
        res=pd.concat([res,inference_results],axis=0)
        
    res.to_csv(proj_dir+'/'+'bounding_boxes.csv',index=False)
