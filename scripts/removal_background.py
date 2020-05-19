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


IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

STRIDE = 32
RESTORE_FROM = './pretrained/indexnet_matting.pth.tar'
RESULT_DIR = './examples/mattes'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    new_x1 = cv2.resize(x1, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

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
        alpha = cv2.resize(outputs, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
        print('framerate: {0:.2f}Hz'.format(running_frame_rate))
        return alpha.astype(np.uint8)


def gen_trimap(mask,k_size=(5,5),ite=1):
    kernel = np.ones(k_size,np.uint8)  # 要素が一の配列を生成する->1画素で処理する周りの画素のサイズ
    eroded = cv2.erode(mask,kernel,iterations = ite) # 収縮
    dilated = cv2.dilate(mask,kernel,iterations = ite) # 膨張
    trimap = np.full(mask.shape,128)
    trimap[eroded > 200] = 255
    trimap[dilated == 0] = 0
    return trimap

def removal_background(img):
    img = img[...,::-1] #BGR->RGB
    h,w,_ = img.shape
    img = cv2.resize(img,(320,320))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval();

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    ret,mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img,(w,h))

    trimap = gen_trimap(mask,k_size=(10,10),ite=5)
    # cv2.imwrite('./examples/trimaps/63.png',trimap) # trimapの書き出し

    # indexnet_mattingの呼び出し
    matte = inference(img, trimap)

    img = img[...,::-1]
    bg = np.full_like(img,255) #white background
    img = img.astype(float)
    bg = bg.astype(float)

    matte = cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)

    matte = matte.astype(float)/255.0

    img = cv2.multiply(img, matte)
    bg = cv2.multiply(bg, 1.0 - matte)
    return cv2.add(img, bg)



if __name__ == "__main__":
    # imageの読み込みなど
    image_path = "./examples/images/062.png"
    img = cv2.imread(image_path)
    outImage = removal_background(img)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(outImage/255)
    cv2.imwrite('./examples/outs/62.png', outImage)
    