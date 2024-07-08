import chainlit as cl
from app.model import Florence2Model
from app.config import ModelConfig
from app.utils import draw_polygons, plot_bbox, draw_ocr_bboxes
from logging_config import get_logger
import io
from PIL import Image
import json
import re

logger = get_logger(__name__)

model = Florence2Model(ModelConfig())

TASK_TYPES = [
    "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>",
    "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>",
    "<OCR>", "<OCR_WITH_REGION>"
]

# State to track user session
user_session = {}

@cl.on_chat_start
async def start():
    logger.debug("on_chat_start function called")
    await cl.Message(content="Welcome! Please select a task type from the list and type it in the chat.").send()
    task_list = "\n".join([f"- {task}" for task in TASK_TYPES])
    await cl.Message(content=f"Available tasks:\n{task_list}").send()
    await cl.Message(content="Please type the task you want to perform and press Enter.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id", None)
    if not session_id:
        session_id = message.id  # Using message.id as a session identifier
        cl.user_session.set("session_id", session_id)

    user_data = user_session.get(session_id, {})
    
    if 'task_type' not in user_data:
        task_type = message.content.strip()
        if task_type not in TASK_TYPES:
            logger.warning(f"Invalid task type: {task_type}")
            await cl.Message(content=f"Invalid task type. Please choose from: {', '.join(TASK_TYPES)}").send()
            return
        user_data['task_type'] = task_type
        user_session[session_id] = user_data
        await cl.Message(content="Please upload an image by clicking the 'Upload' button below.").send()
        return
    
    image = None
    for element in message.elements:
        logger.debug(f"Element received: {type(element)}")
        if isinstance(element, cl.Image):
            image = element
            logger.info(f"Image received: {image}")

    if image:
        user_data['image'] = image
        user_session[session_id] = user_data

        task_type = user_data['task_type']
        if task_type in ["<CAPTION_TO_PHRASE_GROUNDING>", "<REFERRING_EXPRESSION_SEGMENTATION>", "<OPEN_VOCABULARY_DETECTION>"]:
            await cl.Message(content="Please provide the additional text input and press Enter:").send()
            return
        elif task_type in ["<REGION_TO_SEGMENTATION>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
            await cl.Message(content="Please provide region coordinates (e.g., <loc_702><loc_575><loc_866><loc_772>) and press Enter:").send()
            return
        else:
            await process_image(session_id)
    else:
        text_input = message.content.strip()
        user_data['text_input'] = text_input
        user_session[session_id] = user_data
        await process_image(session_id)

def parse_location_string(location_str):
    logger.debug(f"Parsing location string: {location_str}")
    try:
        polygons = []
        current_polygon = []
        
        # Use regex to find all numbers in the string
        numbers = re.findall(r'\d+', location_str)
        
        # Group numbers into coordinate pairs
        for i in range(0, len(numbers), 2):
            if i + 1 < len(numbers):
                x, y = int(numbers[i]), int(numbers[i+1])
                current_polygon.append((x, y))  # Use tuples for coordinates
            
            # Start a new polygon every 50 points (adjust as needed)
            if len(current_polygon) == 50:
                polygons.append(current_polygon)
                current_polygon = []
        
        # Add the last polygon if it's not empty
        if current_polygon:
            polygons.append(current_polygon)
        
        logger.debug(f"Parsed {len(polygons)} polygons")
        logger.debug(f"First polygon: {polygons[0] if polygons else 'No polygons'}")
        return {"polygons": polygons, "labels": [''] * len(polygons)}
    except Exception as e:
        logger.exception(f"Error parsing location string: {str(e)}")
        return None

async def process_image(session_id):
    user_data = user_session.get(session_id, {})
    task_type = user_data['task_type']
    image = user_data['image']
    text_input = user_data.get('text_input', None)

    try:
        logger.debug(f"Calling model.run_example with task_type: {task_type}")
        
        # Load image data from file
        with open(image.path, 'rb') as img_file:
            image_data = img_file.read()
        
        logger.debug(f"Image data loaded, size: {len(image_data)} bytes")
        
        # Ensure the task token is the only token in the prompt
        prompt = task_type
        if text_input and task_type not in text_input:
            prompt += text_input
        
        result = model.run_example(prompt, text_input, image_data)
        logger.debug(f"Raw model output: {result}")
        logger.debug(f"model.run_example completed with result: {result}")

        # Extract the correct key from the result
        expected_key = next((key for key in result.keys() if key.startswith(task_type)), None)
        if not expected_key:
            logger.error(f"Expected key starting with '{task_type}' not found in the result. Available keys: {list(result.keys())}")
            raise KeyError(f"Expected key starting with '{task_type}' not found in the result")

        task_result = result[expected_key]
        logger.debug(f"Task result for '{expected_key}': {task_result}")

        # Check if task_result is a string and needs parsing
        if isinstance(task_result, str):
            logger.debug(f"Task result is a string: {task_result[:100]}...")  # Log first 100 characters
            if task_type in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
                parsed_result = parse_location_string(task_result)
                if parsed_result and parsed_result['polygons']:
                    task_result = parsed_result
                    logger.info(f"Successfully parsed {len(task_result['polygons'])} polygons")
                else:
                    logger.warning(f"Failed to parse location string for task '{task_type}'. Using original string.")
                    task_result = {"polygons": [], "labels": [task_result]}
            else:
                if task_result.startswith('{') or task_result.startswith('['):
                    try:
                        task_result = json.loads(task_result)
                        logger.debug(f"Parsed task result: {task_result}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError: {str(e)}. task_result: {task_result}")
                        task_result = {"polygons": [], "labels": [task_result]}

        # Log the type and content of task_result before drawing polygons
        logger.debug(f"task_result type: {type(task_result)}")
        logger.debug(f"task_result content: {task_result}")

        if task_type in ["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>"]:
            logger.info("Generating bounding box plot")
            fig = plot_bbox(Image.open(image.path), task_result)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            await cl.Message(content=f"Result: {result}", elements=[cl.Image(name="result.png", content=buf.getvalue())]).send()
        elif task_type in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
            logger.info("Drawing polygons")
            original_image = Image.open(image.path).copy()
            output_image = original_image.copy()
            logger.debug(f"Original image size: {output_image.size}")
            logger.debug(f"Number of polygons to draw: {len(task_result['polygons'])}")
            for i, polygon in enumerate(task_result['polygons']):
                logger.debug(f"Polygon {i}: {polygon[:5]}...")  # Log first 5 points of each polygon
            draw_polygons(output_image, task_result, fill_mask=True)
            logger.debug(f"Image after drawing polygons, size: {output_image.size}")
            
            # Save and send both original and processed images
            original_buf = io.BytesIO()
            original_image.save(original_buf, format='PNG')
            original_buf.seek(0)
            
            processed_buf = io.BytesIO()
            output_image.save(processed_buf, format='PNG')
            processed_buf.seek(0)
            
            await cl.Message(content=f"Result: Detected {len(task_result['polygons'])} regions", elements=[
                cl.Image(name="original.png", content=original_buf.getvalue(), display="inline"),
                cl.Image(name="processed.png", content=processed_buf.getvalue(), display="inline")
            ]).send()
            logger.info(f"Sent message with {len(task_result['polygons'])} regions")
        elif task_type == "<OCR_WITH_REGION>":
            logger.info("Drawing OCR bounding boxes")
            logger.info(f"Number of OCR boxes: {len(task_result['quad_boxes'])}")
            logger.info(f"OCR labels: {task_result['labels']}")
            output_image = Image.open(image.path).copy()
            output_image = model.preprocess_image(output_image)  # Preprocess the image
            draw_ocr_bboxes(output_image, task_result)
            buf = io.BytesIO()
            output_image.save(buf, format='PNG')
            buf.seek(0)
            await cl.Message(content=f"Result: {result}", elements=[cl.Image(name="result.png", content=buf.getvalue())]).send()
        else:
            await cl.Message(content=f"Result: {result}").send()
        
        # Clear session after processing
        user_session.pop(session_id, None)
    except Exception as e:
        logger.exception(f"Error in process_image function: {str(e)}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()
        user_session.pop(session_id, None)

if __name__ == "__main__":
    logger.debug("Starting Chainlit application")
    cl.run()