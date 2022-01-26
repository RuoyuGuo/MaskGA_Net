import os

from img_norm.norm1 import main as main1

def norm(source_image_dir, result_dir, method):

    template_dir = os.path.join('.', 'img_norm', str(method), 'data', 'template')

    os.makedirs(result_dir, exist_ok=True)
    
    main1.norm(source_image_dir, template_dir, result_dir)
