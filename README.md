# Cloth Segmentation and Color Modification

## Revamp Your Wardrobe with AI: Seamless Cloth Segmentation and Custom Colorization

### Project Explanation:
Experience the future of fashion design with our AI-powered application that transforms how you visualize and customize clothing. Our advanced cloth segmentation technology precisely identifies and isolates different garment sections in any image. Using a user-friendly interface, you can effortlessly apply custom colors to each segment, experiment with suggested palettes, and instantly see your design come to life. Whether you're a designer seeking inspiration or simply want to try new styles, our tool offers endless possibilities for personalization and creativity.

## Features
- **Home Page:** Overview of the project with a marketing title and explanation.
- **U2Net Segmentation:** Upload an image and generate segmentation masks using the U2Net model.
- **Unet Segmentation:** Upload an image and generate segmentation masks using the Unet model.
- **Customize Class Colors:** Choose custom colors and transparency levels for each segment.
- **Suggested Images:** Generate and display suggested images based on suitable color palettes.

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Dependencies
Install the required packages using pip:
```sh
pip install -r requirements.txt
```

### requirements.txt
```
streamlit
numpy
pillow
opencv-python
torch
torchvision
albumentations
```

## Usage

### Running the Application
To run the Streamlit application, use the following command:
```sh
streamlit run app.py
```

### Navigation
- **Home:** Provides an overview of the project.
- **U2Net Segmentation:** Allows you to upload an image and generate segmentation masks using the U2Net model. You can customize class colors and view suggested images based on suitable color palettes.
- **Unet Segmentation:** Allows you to upload an image and generate segmentation masks using the Unet model. You can download the mask and blended images.

### Customize Class Colors
- Upload an image.
- Select the number of colors to extract per category.
- Choose the color palette JSON file path and model checkpoint path.
- Use CUDA if available.
- Customize the colors and transparency levels for each segment.
- Download the customized image.

### Suggested Images
- View and download suggested images based on suitable color palettes.

## File Structure
```
├── app.py                  # Main Streamlit application file
├── process.py              # Contains functions for segmentation and color application
├── network.py              # Network architecture for U2Net (not provided here, you need to include it)
├── requirements.txt        # Dependencies required for the project
├── 100_five_color_palettes.json  # JSON file with predefined color palettes
└── output/                 # Directory to save the segmentation masks and results
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgments
- The U2Net and Unet models used in this project are credited to their respective authors and repositories. For more information about U2Net, visit [this link](https://github.com/levindabhi/cloth-segmentation?tab=readme-ov-file).
- Special thanks to the developers of Streamlit for providing an easy-to-use framework for building web applications in Python.
