# AnimeFaceGenWGANGP

AnimeFaceGenWGANGP is a deep learning project for generating anime-style faces using a Wasserstein GAN with Gradient Penalty (WGAN-GP). This repository contains the code, pre-trained models, and instructions to generate new anime faces and train your own generator.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Files](#model-files)
- [Training](#training)

## Overview

This project implements a WGAN-GP architecture to generate high-quality anime faces. The generator and discriminator are defined in `generator_model.py`, and pre-trained weights are provided for quick generation.

## Features

- Generate anime faces using pre-trained models
- Train your own generator with custom datasets
- Modular code for easy experimentation

## Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/raogaurav17/anime_face_generator_wgangp.git
   cd anime_face_generator_wgangp
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Generate Anime Faces

Run the main application to generate faces using the pre-trained generator:

```powershell
python app.py
```

You can also use `Script.py` for custom generation or testing.

### Model Files

- `best_geenerator.pth`: Pre-trained generator weights.
- Place your desired model file in the project root and update the path in `app.py` or `Script.py` if needed.

### Training

To train your own generator, follow these steps:

1. Download and extract the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) into your project directory.
2. Open `generator_model.py` and adjust the dataset path, training parameters (epochs, batch size, learning rate), and model architecture as needed.
3. Run the training script:
   ```powershell
   python generator_model.py
   ```
4. The trained model weights will be saved in the project directory. You can rename or move them as needed for generation.

#### Deploying the Model

To deploy and use the trained generator:

1. Place the trained `.pth` file (e.g., `best_generator.pth`) in the project root.
2. Update the model path in `app.py` or `Script.py` if you are using a custom filename.
3. Run the generation script to produce new anime faces:
   ```powershell
   python app.py
   ```
4. For web or API deployment, wrap the generation logic in a Flask or FastAPI app and expose endpoints for image generation. (See Python web frameworks for details.)

#### Dataset Used

This project uses the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle for training. The dataset contains thousands of cropped anime face images suitable for generative modeling. Download the dataset from Kaggle and place it in your project directory or specify the path in your training script.

## Results

Sample generated anime faces will be saved in the output directory or displayed after running the script. You can further customize the output location in the code.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## Project Structure

```
├── app.py                # Application for face generation interface using Gradio
├── Script.py             # Script for Model training
├── generator_model.py    # Model architecture and training logic
├── best_generator.pth    # Pre-trained generator weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── __pycache__/          # Python cache files
```

## Contact

For questions, suggestions, or collaboration, please contact the repository owner:

- GitHub: [raogaurav17](https://github.com/raogaurav17)
- LinkedIn: [ydv17gaurav](https://www.linkedin.com/in/ydv17gaurav)
