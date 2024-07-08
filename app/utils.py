import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import random
import logging
import numpy as np
from matplotlib.colors import to_rgba

logger = logging.getLogger(__name__)

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    plt.close(fig)
    return fig

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    logger.info(f"Drawing {len(prediction['polygons'])} polygons")

    for i, polygon in enumerate(prediction['polygons']):
        if len(polygon) < 3:
            logger.warning(f"Skipping polygon {i} because it has less than 3 points: {polygon}")
            continue

        color = random.choice(colormap)  # Random RGB color
        fill_color = color if fill_mask else None  # Color for fill if needed

        try:
            # Log raw coordinates for debugging
            logger.debug(f"Polygon {i} raw coordinates: {polygon}")

            # Convert all coordinates to integers and ensure they're within image boundaries
            scaled_polygon = [(max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))) for x, y in polygon]

            # Log scaled coordinates for debugging
            logger.debug(f"Polygon {i} scaled coordinates: {scaled_polygon}")

            # Draw the raw coordinates as small points for visualization
            for x, y in polygon:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline="yellow", fill="yellow")

            # Draw the polygon
            draw.polygon(scaled_polygon, outline=color, fill=fill_color)

            # Draw each point of the polygon
            for point in scaled_polygon:
                draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=color)

            logger.debug(f"Drew polygon {i} with {len(scaled_polygon)} points: {scaled_polygon[:5]}...")  # Log first 5 points

            # Draw the label
            x, y = scaled_polygon[0]
            draw.text((x, y), f"{i + 1}", fill=color)  # Added index for easier identification
        except Exception as e:
            logger.error(f"Error drawing polygon {i}: {str(e)}")
            logger.error(f"Polygon data: {polygon}")

    logger.info("Finished drawing polygons")
    return image

def draw_ocr_bboxes(image, prediction):
    draw = ImageDraw.Draw(image)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    
    num_boxes = len(prediction['quad_boxes'])
    logger.info(f"Number of OCR boxes detected: {num_boxes}")

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(prediction['quad_boxes'], prediction['labels'])):
        color = colors[i % len(colors)]
        box = np.array(box).reshape(-1, 2)
        draw.polygon(box.flatten().tolist(), outline=color, width=2)
        draw.text((box[0][0], box[0][1]), f"{i+1}:{label[:10]}", fill=color, font=font)
    
    return image