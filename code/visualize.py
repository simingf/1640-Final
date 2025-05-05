#Get 10 test examples 
#Load weights 

#For each test example...
    #Forward pass of model 
    #Argmax to find degree of most confidence, save as variable
    #Use that degree to rotate the test example image 
    #Save the predicted image, rotation prediction (#), and ground truth into visualization_preds (write the rotation degrees to a txt)


"""
visualize.py
------------
* Loads a trained SimpleModel (weights produced by train.py)
* Grabs 10 samples from the test set
* Predicts the rotation for each sample
* Rotates the image by the predicted angle
* Saves the rotated image plus a text file with
  predicted vs. ground‑truth degrees in  ./visualization_preds
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

from dataloader import get_test_dataloader
from model import ResNetModel


# ---------- helper utilities --------------------------------------------------
def tensor_to_pil(img_tensor):
    """
    img_tensor: (C, H, W) scaled to [0,1] or [-1,1].
    Returns: PIL.Image in RGB for easy saving / rotation.
    """
    t = img_tensor.detach().cpu()

    # If you normalised to mean/std in your dataloader, undo it here
    # (This assumes you *didn't* normalise.  Adapt as needed.)
    t = t.clamp(0, 1)

    t = t.mul(255).byte()
    t = t.permute(1, 2, 0)          # C,H,W  ➜  H,W,C
    return Image.fromarray(t.numpy())

def circle_mask(image):
    """
    Apply a circular mask to the image.
    Args:
        image (PIL.Image): The input image.
    Returns:
        PIL.Image: The image with the circular mask applied.
    """
    width, height = image.size
    assert width == height, "Image must be square for circular mask."
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    image.putalpha(mask)
    return image

# ---------- main visualisation script ----------------------------------------
def main():
    # --- configuration --------------------------------------------------------
    NUM_EXAMPLES      = 5
    IMAGE_SIZE        = (224, 224)            # must match train.py
    WEIGHTS_PATH      = Path("../model_weights/resnet18.pth")
    OUT_DIR           = Path("../visualization180")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TXT_PATH          = OUT_DIR / "predictions.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # --- data -----------------------------------------------------------------
    test_loader = get_test_dataloader(num_images=2000, image_size=IMAGE_SIZE)

    # --- model ----------------------------------------------------------------
    model = ResNetModel().to(device)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find weight file at {WEIGHTS_PATH}. "
            "Train the model first or update WEIGHTS_PATH."
        )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # --- iterate over the first NUM_EXAMPLES test images ----------------------
    written = 0
    with TXT_PATH.open("w") as txt, torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds   = torch.argmax(outputs, dim=1)    # shape: (batch,)
            targets = torch.argmax(targets, dim=1)

            for i in range(inputs.size(0)):
                if written >= NUM_EXAMPLES:
                    break

                pred_deg = int(preds[i].item())
                gt_deg   = int(targets[i].item())

                if abs(pred_deg - gt_deg) == 180:
                    captcha = tensor_to_pil(inputs[i])
                    captcha = circle_mask(captcha)
                    captcha_file = OUT_DIR / f"sample_{written:02d}_captcha.png"
                    captcha.save(captcha_file)

                    sol_image = TF.rotate(captcha, angle=-gt_deg, fill=0)
                    sol_image = circle_mask(sol_image)
                    sol_file = OUT_DIR / f"sample_{written:02d}_solution.png"
                    sol_image.save(sol_file)

                    predicted_image = TF.rotate(captcha, angle=-pred_deg, fill=0)
                    predicted_image = circle_mask(predicted_image)
                    predicted_file = OUT_DIR / f"sample_{written:02d}_predicted.png"
                    predicted_image.save(predicted_file)

                    txt.write(
                        f"sample_{written:02d}: predicted = {pred_deg:3d}°, "
                        f"ground‑truth = {gt_deg:3d}°\n"
                    )
                    written += 1

            if written >= NUM_EXAMPLES:
                break

    print(f"✅  Saved {written} examples to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()