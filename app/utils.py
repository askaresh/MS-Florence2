import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import random
import logging
import numpy as np
from matplotlib.colors import to_rgba
import io

logger = logging.getLogger(__name__)

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    try:
        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1-5, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
    except TypeError as e:
        logger.error(f"TypeError in plot_bbox: {str(e)}")
        logger.error(f"Data received: {data}")

    ax.axis('off')
    plt.tight_layout()
    plt.close(fig)
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    logger.info(f"Image size: {width}x{height}")

    polygons = prediction['polygons']
    labels = prediction['labels']

    if isinstance(polygons[0], list) and isinstance(polygons[0][0], list):
        polygons = polygons[0]

    color = (255, 0, 0, 128)  # Red with 50% opacity
    outline_color = (255, 0, 0)  # Solid red for outline

    try:
        scaled_polygon = [(max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))) 
                          for x, y in zip(polygons[0][::2], polygons[0][1::2])]
        
        if len(scaled_polygon) > 2:
            if fill_mask:
                draw.polygon(scaled_polygon, fill=color, outline=outline_color)
            else:
                draw.line(scaled_polygon + [scaled_polygon[0]], fill=outline_color, width=3)

        # Add a visible marker to confirm drawing occurred
        draw.text((10, 10), "Segmentation applied", fill=(255, 0, 0))

        logger.info(f"Drew polygon with {len(scaled_polygon)} points")
    except Exception as e:
        logger.error(f"Error drawing polygon: {str(e)}")

    return image

def draw_ocr_bboxes(image, prediction):
    draw = ImageDraw.Draw(image)
    
    num_boxes = len(prediction['quad_boxes'])
    logger.info(f"Number of OCR boxes detected: {num_boxes}")

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(prediction['quad_boxes'], prediction['labels'])):
        color = random.choice(colormap)
        box = np.array(box).reshape(-1, 2)
        draw.polygon(box.flatten().tolist(), outline=color, width=2)
        draw.text((box[0][0], box[0][1]-20), f"{label[:10]}", fill=color, font=font)
    
    return image

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)