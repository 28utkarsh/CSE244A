import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from timm import create_model
import pandas as pd
from PIL import Image

class TestDataset(Dataset):
    """
    Custom Dataset for loading test images.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)

def load_checkpoint(checkpoint_path, model):
    """
    Load model weights from a checkpoint.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully.")

def main():
    # Paths
    checkpoint_path = 'Checkpoint/best_checkpoint.pth'  # Replace with your checkpoint path
    test_data_path = 'test'        # Replace with your test dataset directory
    output_csv_path = 'predictions.csv'         # Output CSV file
    batch_size = 128
    num_workers = 16

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = create_model('deit3_large_patch16_384', pretrained=False, num_classes=135)
    model.to(device)
    model.eval()

    # Load checkpoint
    load_checkpoint(checkpoint_path, model)

    # Test dataset and dataloader
    transform_test = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = TestDataset(image_dir=test_data_path, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Inference
    predictions = []
    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            for image_name, label in zip(image_names, predicted.cpu().numpy()):
                predictions.append((image_name, label))

    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=['image', 'id'])
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == '__main__':
    main()

