import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from time import time
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from hlmobilenetv2 import hlmobilenetv2

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

STRIDE = 32
RESTORE_FROM = './pretrained/indexnet_matting.pth.tar'
RESULT_DIR = './examples/mattes'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# load pretrained model
net = hlmobilenetv2(
    pretrained=False,
    freeze_bn=True,
    output_stride=STRIDE,
    apply_aspp=True,
    conv_operator='std_conv',
    decoder='indexnet',
    decoder_kernel_size=5,
    indexnet='depthwise',
    index_mode='m2o',
    use_nonlinear=True,
    use_context=True
)

try:
    checkpoint = torch.load(RESTORE_FROM, map_location=device)
    pretrained_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            key = key[7:]
        pretrained_dict[key] = value
except:
    raise Exception('Please download the pretrained model!')
net.load_state_dict(pretrained_dict)
net.to(device)
if torch.cuda.is_available():
    net = nn.DataParallel(net)

# switch to eval mode
net.eval()


def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x


def inference(image, trimap):
    with torch.no_grad():
        # image, trimap = read_image(image_path), read_image(trimap_path)
        trimap = np.expand_dims(trimap, axis=2)
        image = np.concatenate((image, trimap), axis=2)

        h, w = image.shape[:2]

        image = image.astype('float32')
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        image = image.astype('float32')

        image = image_alignment(image, STRIDE)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        inputs = inputs.to(device)

        # inference
        start = time()
        outputs = net(inputs)
        end = time()

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv2.resize(outputs, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        running_frame_rate = 1 * float(1 / (end - start))  # batch_size = 1
        # print('framerate: {0:.2f}Hz'.format((end - start)*1000.))
        return alpha.astype(np.uint8)


def gen_trimap(mask, k_size=(5, 5), ite=1):
    kernel = np.ones(k_size, np.uint8)  # 要素が一の配列を生成する->1画素で処理する周りの画素のサイズ
    eroded = cv2.erode(mask, kernel, iterations=ite)  # 収縮
    dilated = cv2.dilate(mask, kernel, iterations=ite)  # 膨張
    trimap = np.full(mask.shape, 128)
    trimap[eroded > 200] = 255
    trimap[dilated == 0] = 0
    return trimap


def removal_background(input_img, bg):
    t1 = time() * 1000.
    original = np.copy(input_img)
    img = input_img[..., ::-1]  # BGR->RGB
    h, w, _ = img.shape
    img = cv2.resize(img, (320, 320))

    t2 = time() * 1000.

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    trimap = gen_trimap(mask, k_size=(10, 10), ite=5)
    t3 = time() * 1000.
    # cv2.imwrite('./examples/trimaps/63.png',trimap) # trimapの書き出し

    # indexnet_mattingの呼び出し
    matte = inference(original, trimap)

    # white background
    original = original.astype(float)

    matte = cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)  # 10~12ms

    matte = matte.astype(float) / 255.0
    t4 = time() * 1000.
    original = cv2.multiply(original, matte)  # 18~20ms
    bg_removed = cv2.multiply(bg, 1.0 - matte)  # 10~20ms
    t5 = time() * 1000.

    out =  cv2.add(original, bg_removed)

    t6 = time() * 1000.
    # print("{}, {}, {}, {} {} {}".format(t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6-t1))
    return out


if __name__ == "__main__":
    # imageの読み込みなど
    image_path = "./examples/images/062.png"
    img = cv2.imread(image_path)

    bg = np.full_like(img, 255)
    bg = bg.astype(float)

    outImage = removal_background(img, bg)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(outImage/255)
    cv2.imwrite('./examples/outs/62.png', outImage)
