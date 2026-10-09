# Neural Style Transfer

Neural Style Transfer is a deep learning project that combines the content of one image with the artistic style of another to generate a stylized image. Built using PyTorch and VGG19, the project uses Adaptive Instance Normalization (AdaIN) to transfer visual styles while preserving the structure of the original image.

## Features

- Transfer artistic styles onto content images
- VGG19-based feature extraction
- Adaptive Instance Normalization (AdaIN) for style transfer
- Image processing and generation using PyTorch
- Pretrained model checkpoints for inference

## Dataset

The project uses a dataset of **19,501 images** for training and experimentation with neural style transfer, providing a collection of visual references for working with different artistic styles.

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Model Architecture:** VGG19
- **Style Transfer:** Adaptive Instance Normalization (AdaIN)

## How It Works

1. Upload a content image and a reference style image.
2. Extract feature representations from both images using a pretrained VGG19 network.
3. Apply Adaptive Instance Normalization to align the content features with the style image's feature statistics.
4. Decode the transformed features to generate the stylized image.

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/MADHAVENDRA-maker/NST_PROJECT_Github.git
cd NST_PROJECT_Github
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Ensure that the required pretrained model weights are available at the paths expected by the application.

## Author

**Madhavendra Gautam**  
Indian Institute of Technology Jammu

Interested in deep learning, computer vision, and generative AI.
