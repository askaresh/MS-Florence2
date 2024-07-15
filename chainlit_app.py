import chainlit as cl
from app.model import Florence2Model
from app.config import ModelConfig
from logging_config import get_logger
import io
from PIL import Image
from app.utils import draw_polygons, plot_bbox, draw_ocr_bboxes, fig_to_pil

logger = get_logger(__name__)

model = Florence2Model(ModelConfig())

TASK_TYPES = [
    "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>",
    "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>",
    "<OCR>", "<OCR_WITH_REGION>"
]

user_session = {}

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Please select a task type from the list.").send()
    task_list = "\n".join([f"- {task}" for task in TASK_TYPES])
    await cl.Message(content=f"Available tasks:\n{task_list}").send()

@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id", message.id)
    cl.user_session.set("session_id", session_id)

    user_data = user_session.get(session_id, {})
    
    if 'task_type' not in user_data:
        task_type = message.content.strip()
        if task_type not in TASK_TYPES:
            await cl.Message(content=f"Invalid task type. Please choose from: {', '.join(TASK_TYPES)}").send()
            return
        user_data['task_type'] = task_type
        user_session[session_id] = user_data
        await cl.Message(content="Please upload an image.").send()
        return
    
    image = next((element for element in message.elements if isinstance(element, cl.Image)), None)

    if image:
        user_data['image'] = image
        user_session[session_id] = user_data

        task_type = user_data['task_type']
        if task_type in ["<CAPTION_TO_PHRASE_GROUNDING>", "<REFERRING_EXPRESSION_SEGMENTATION>", "<OPEN_VOCABULARY_DETECTION>"]:
            await cl.Message(content="Please provide the additional text input:").send()
            return
        else:
            await process_image(session_id)
    else:
        text_input = message.content.strip()
        user_data['text_input'] = text_input
        user_session[session_id] = user_data
        await process_image(session_id)

async def process_image(session_id):
    user_data = user_session.get(session_id, {})
    task_type = user_data['task_type']
    image = user_data['image']
    text_input = user_data.get('text_input')

    try:
        with open(image.path, 'rb') as img_file:
            image_data = img_file.read()
        
        original_image = Image.open(image.path)
        result = model.run_example(task_type, text_input, image_data)
        
        if task_type in ["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>"]:
            fig = plot_bbox(original_image, result[task_type])
            output_image = fig_to_pil(fig)
            
            original_buf = io.BytesIO()
            original_image.save(original_buf, format='PNG')
            original_buf.seek(0)
            
            output_buf = io.BytesIO()
            output_image.save(output_buf, format='PNG')
            output_buf.seek(0)
            
            await cl.Message(content=f"Result: {result}", elements=[
                cl.Image(name="original.png", content=original_buf.getvalue(), display="inline"),
                cl.Image(name="result.png", content=output_buf.getvalue(), display="inline")
            ]).send()
            
        elif task_type in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
            output_image = original_image.copy()
            segmentation_result = result[task_type]
            draw_polygons(output_image, segmentation_result, fill_mask=True)
            
            original_buf = io.BytesIO()
            original_image.save(original_buf, format='PNG')
            original_buf.seek(0)
            
            output_buf = io.BytesIO()
            output_image.save(output_buf, format='PNG')
            output_buf.seek(0)
            
            logger.info("Segmentation image saved to buffer")
            await cl.Message(content="Segmentation complete", elements=[
                cl.Image(name="original.png", content=original_buf.getvalue(), display="inline"),
                cl.Image(name="segmented.png", content=output_buf.getvalue(), display="inline")
            ]).send()
            logger.info("Segmentation message sent")
            
        elif task_type == "<OCR_WITH_REGION>":
            output_image = original_image.copy()
            draw_ocr_bboxes(output_image, result[task_type])
            
            original_buf = io.BytesIO()
            original_image.save(original_buf, format='PNG')
            original_buf.seek(0)
            
            output_buf = io.BytesIO()
            output_image.save(output_buf, format='PNG')
            output_buf.seek(0)
            
            await cl.Message(content=f"Result: {result}", elements=[
                cl.Image(name="original.png", content=original_buf.getvalue(), display="inline"),
                cl.Image(name="result.png", content=output_buf.getvalue(), display="inline")
            ]).send()
            
        else:
            await cl.Message(content=f"Result: {result}").send()
        
        logger.info(f"{task_type} task completed and message sent")
        user_session.pop(session_id, None)
    except Exception as e:
        logger.exception(f"Error in process_image function: {str(e)}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()
        user_session.pop(session_id, None)

if __name__ == "__main__":
    cl.run()