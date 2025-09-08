# SEMClassify

SEMClassify is a project aimed at classifying Scanning Electron Microscope (SEM) images using machine learning techniques. This repository contains code and notebooks to preprocess SEM image data, train classification models, and evaluate their performance.

## Features

- **Image Preprocessing:** Prepare SEM images for model training.
- **Model Training:** Train machine learning models for image classification.
- **Evaluation:** Assess the accuracy and effectiveness of models.
- **Jupyter Notebooks:** Interactive notebooks for experimentation and analysis.

## Technologies Used

- **Jupyter Notebook** (98.3%)
- **Python** (1.7%)
- Popular ML libraries: `scikit-learn`, `TensorFlow`, `Keras`, `matplotlib`, `numpy`, etc.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Package manager: pip or conda

### Installation

Clone the repository:

```bash
git clone https://github.com/senghuyjr11/SEMClassify.git
cd SEMClassify
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

2. Open the provided notebooks and follow the instructions to run experiments.

### Data

- Place your SEM images in the `data/` directory.
- Update notebook paths if needed.

## Project Structure

```
SEMClassify/
│
├── notebooks/        # Jupyter notebooks for experiments
├── src/              # Python scripts for preprocessing and modeling
├── data/             # SEM image datasets
├── requirements.txt  # Python dependencies
└── README.md         # Project overview
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

## Contact

Maintainer: [senghuyjr11](https://github.com/senghuyjr11)

Feel free to reach out with questions, suggestions, or feedback!

yolo detect train model=yolov8n.pt data=yolo_corrosion.yaml epochs=50 imgsz=640 project=model name=corrosion_yolov8

yolo detect val model=model/corrosion_yolov8/weights/best.pt data=yolo_corrosion.yaml project=model name=val_results

yolo detect predict model=model/corrosion_yolov8/weights/best.pt source=some_folder_or_image project=model name=inference
