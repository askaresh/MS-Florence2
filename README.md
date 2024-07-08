# Comprehensive Image Analysis API with Microsoft Florence-2-large, Chainlit, and Docker

Welcome to our project where we build a comprehensive Image Analysis API using Microsoft Florence-2-large, Chainlit, and Docker. This API allows you to perform various image analysis tasks such as captioning, object detection, expression segmentation, and OCR.

## My Blog Link
Link - https://askaresh.com/2024/07/08/building-a-comprehensive-image-analysis-api-with-microsoft-florence-2-large-chainlit-and-docker/


## Model Overview

We utilized the pre-trained Florence-2-large model from Hugging Face Transformers for image analysis. This powerful model has been trained on a vast dataset of images and can perform multiple tasks such as image captioning, object detection, and OCR.


## Implementation Details

- **chainlit_app.py**: This is the heart of our Chainlit application. It defines the message handler that processes uploaded images and generates responses using the Florence model.
- **app/model.py**: Contains the `ModelManager` class, responsible for loading and managing the Florence-2-large model.
- **app/utils.py**: Contains utility functions for image drawing, plot boxes, polygons, and OCR boxes.
- **logging_config.py**: Manages detailed logging for the project.
- **Dockerfile**: Defines how our application is containerized, ensuring all dependencies are properly installed and the environment is consistent.

## Task Prompts

We defined several task prompts to leverage the Florence-2-large model's capabilities. These prompts include:

- `<CAPTION>`: Generates a simple caption for the image.
- `<DETAILED_CAPTION>`: Provides a detailed description of the image.
- `<OD>`: Detects and locates objects within the image.
- `<OCR>`: Performs Optical Character Recognition.
- `<CAPTION_TO_PHRASE_GROUNDING>`: Locates specific phrases or objects mentioned in the caption within the image.
- `<DENSE_REGION_CAPTION>`: Generates captions for specific regions within the image.
- `<REGION_PROPOSAL>`: Suggests regions of interest within the image without labeling them.
- `<MORE_DETAILED_CAPTION>`: Generates a very comprehensive description of the image.
- `<REFERRING_EXPRESSION_SEGMENTATION>`: Segments the image based on a textual description of a specific object or region.
- `<REGION_TO_SEGMENTATION>`: Generates a segmentation mask for a specified region in the image.
- `<OPEN_VOCABULARY_DETECTION>`: Detects objects in the image based on user-specified categories.
- `<REGION_TO_CATEGORY>`: Classifies a specific region of the image into a category.
- `<REGION_TO_DESCRIPTION>`: Generates a detailed description of a specific region in the image.
- `<OCR_WITH_REGION>`: OCR on specific regions of the image.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MS-FLORENCE2.git
   cd MS-FLORENCE2

2. Create a .env file with the following content:
    env
    Copy code
    MODEL_ID=microsoft/Florence-2-large
    RATE_LIMIT=5

3. Build the Docker container:
    docker build -t florence2-image-analysis .

4. Run the Docker container:
    docker run --gpus all -p 8010:8010 florence2-image-analysis

## Running the Application
Once the container is running, access the Chainlit interface at http://localhost:8010. Follow the instructions in the chat to perform various image analysis tasks.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.