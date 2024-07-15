import torch
from PIL import Image
import io
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class Florence2Model:
    def __init__(self, config):
        logger.debug("Florence2Model.__init__ called")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, trust_remote_code=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        logger.info("Model loaded successfully")

    def preprocess_image(self, image):
        logger.debug("Preprocessing image")
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        return image

    def run_example(self, task_prompt, text_input=None, image_data=None):
        logger.debug(f"run_example called with task_prompt: {task_prompt}")
        try:
            image = self.preprocess_image(image_data)
            logger.info(f"Image preprocessed, size: {image.size}")

            prompt = task_prompt if text_input is None else task_prompt + text_input
            logger.info(f"Prompt: {prompt}")

            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            logger.info(f"Parsed answer: {parsed_answer}")
            return parsed_answer
        except Exception as e:
            logger.exception(f"Error in run_example: {str(e)}")
            raise