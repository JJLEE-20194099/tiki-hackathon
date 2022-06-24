from webbrowser import BackgroundBrowser
from PIL import Image, ImageDraw, ImageFilter

def draw_img_circle_background(img):
    mask_im = Image.new("L", img.size, 0,)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse([(10, 10), (img.size[0] - 10, img.size[1] - 10)], fill=255)
    mask_im.save('images/mask_circle.jpg', quality=95)
    return mask_im

def combine_outfit_to_image(background_path, outfit, suggestion_folder, file_path):
    res = []
    for cloth in outfit:
        # img_path = cloth["image_path"]
        img_path = cloth
        res.append(Image.open(img_path).resize((200, 200)))
    
    background = Image.open(background_path)
    copy_background = background.copy()
    w, h = copy_background.size
    pos = [(10, 0), (int(w / 2 - 100), 0), (w - 210, 0), (10, h - 210), (int(w / 2 - 100), h - 210), (w - 210, h - 210)]
    for i, img in enumerate(res):
        mask_im = draw_img_circle_background(img)
        copy_background.paste(img, pos[i], mask_im)
        copy_background.save('{}/{}'.format(suggestion_folder, file_path), quality=95)