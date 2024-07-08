from logging_config import get_logger
import os
import time
import io
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageEnhance
import requests
import torch
import numpy as np

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5

class Florence2Model:
    def __init__(self, config):
        logger.debug("Florence2Model.__init__ called")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempting to load model (Attempt {attempt + 1}/{MAX_RETRIES})")
                self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, trust_remote_code=True).eval()
                self.processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
                self.model.to(self.device)
                logger.info("Model loaded successfully")
                break
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Failed to load the model after all attempts")
                    raise

    def preprocess_image(self, image):
        logger.debug("Preprocessing image")
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.debug("Converted image to RGB")
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        logger.debug("Enhanced image contrast")
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        logger.debug("Enhanced image sharpness")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Ensure image has 3 dimensions (height, width, channels)
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=-1)
            image_np = np.repeat(image_np, 3, axis=-1)
            logger.debug("Expanded grayscale image to 3 channels")
        elif len(image_np.shape) == 3 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis[-1])
            logger.debug("Repeated single-channel image to 3 channels")
        elif len(image_np.shape) == 3 and image_np.shape[-1] == 4:
            # If RGBA, remove alpha channel
            image_np = image_np[:, :, :3]
            logger.debug("Removed alpha channel from RGBA image")
        
        # Convert back to PIL Image
        preprocessed_image = Image.fromarray(image_np)
        logger.debug(f"Preprocessed image size: {preprocessed_image.size}")
        return preprocessed_image

    def run_example(self, task_prompt, text_input=None, image_data=None, region=None):
        logger.debug(f"run_example called with task_prompt: {task_prompt}")
        try:
            if image_data is None:
                raise ValueError("Image data is required")

            logger.debug(f"Image data type: {type(image_data)}, size: {len(image_data)} bytes")
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Image loaded, size: {image.size}")

            # Preprocess the image
            image = self.preprocess_image(image)
            logger.info(f"Image preprocessed, new size: {image.size}")

            if text_input is None:
                prompt = task_prompt
            else:
                prompt = task_prompt + text_input
            logger.info(f"Prompt: {prompt}")

            if region:
                prompt += f"<loc_{region[0]}><loc_{region[1]}><loc_{region[2]}><loc_{region[3]}>"
                logger.info(f"Region added to prompt: {prompt}")

            logger.debug("Processing inputs")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            logger.debug(f"Inputs processed: {inputs.keys()}")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.debug("Inputs moved to device")

            logger.debug("Generating output")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_beams=3,
            )
            logger.debug("Output generated")

            logger.info("Decoding generated ids")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            logger.info("Post-processing generation")
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            logger.info("Example run completed successfully")
            return parsed_answer
        except Exception as e:
            logger.exception(f"Error in run_example: {str(e)}")
            raise

    def run_cascaded_task(self, first_task, second_task, image_data):
        logger.info(f"Running cascaded task: {first_task} -> {second_task}")
        try:
            first_result = self.run_example(first_task, image_data=image_data)
            logger.info(f"First task result: {first_result}")
            second_result = self.run_example(second_task, text_input=first_result[first_task], image_data=image_data)
            logger.info(f"Second task result: {second_result}")
            return {first_task: first_result[first_task], second_task: second_result[second_task]}
        except Exception as e:
            logger.error(f"Error in run_cascaded_task: {str(e)}")
            raise
