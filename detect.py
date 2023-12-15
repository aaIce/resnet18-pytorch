import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def preprocess_input(image):
    image /= 255.0
    return image

def resize_img(image, size, letterbox_image=False):
    w, h = size
    iw, ih = image.size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def detect(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using {} device".format(device))
    if opt.weights[-2:] == "pt":
        model = torch.load(opt.weights, map_location=device)
        img = Image.open(opt.source)
        w, h = img.size
        img = img.convert('RGB')
        image_data = resize_img(img, (224, 224))
        # h, w, 3 => 3, h, w => 1, 3, h, w        括号里面的操作相当于transformers.Totensor
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(device)
            pred = model(images)
            _, preds = torch.max(pred, 1)

        font = ImageFont.truetype(
            font='simhei.ttf',
            size=np.floor(3e-2 * max(w, h) + 0.5).astype('int32')
        )
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), f"cls:{preds},conf:{_}", fill=(255, 0, 0), font=font)
        del draw
        img.show()
        img.save('textimage.jpg')
    else:
        print("load model error!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model.pt', help='initial weights path')
    parser.add_argument('--source', type=str, default='flower_data/train/1/image_06734.jpg', help='source')
    parser.add_argument('--img-size', nargs='+', type=int, default=[224, 224], help='[train, test] image sizes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    detect(opt)


