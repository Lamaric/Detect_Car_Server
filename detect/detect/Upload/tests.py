from django.test import TestCase
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib, time
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib, time, os
from fastai.vision import *

colors = plt.cm.hsv(np.linspace(0, 1, 7)).tolist()
labels = ['Scrape','Scratch','Deformation', 'Cracking','Damaged','SeriousDamageSevere']

def combine(img1 = 'img.jpg', img2 = 'img2.jpg'):

    img1 = cv.imread(img1)
    img2 = cv.imread(img2)
    img2 = img2[: 512, :512, :]

    dst = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
    return dst



def test():
    predict = [{'bbox': [324.9660949707031, 390.3648986816406, 67.09695434570312, 10.602996826171875], 'score': 0.26989877223968506, 'category_id': 1}, {'bbox': [309.7373962402344, 367.67327880859375, 22.21588134765625, 34.48779296875], 'score': 0.10203982144594193, 'category_id': 1}, {'bbox': [270.58355712890625, 290.7264709472656, 63.381439208984375, 31.444488525390625], 'score': 0.7487391829490662, 'category_id': 1}, {'bbox': [311.23748779296875, 374.9159240722656, 77.17437744140625, 27.13165283203125], 'score': 0.08209992200136185, 'category_id': 1}, {'bbox': [304.2052917480469, 326.1581115722656, 38.933502197265625, 79.14605712890625], 'score': 0.1231551542878151, 'category_id': 1}, {'bbox': [260.7698669433594, 290.7528076171875, 96.87548828125, 120.87936401367188], 'score': 0.9802087545394897, 'category_id': 1}, {'bbox': [284.822509765625, 305.2769775390625, 100.84872436523438, 96.3428955078125], 'score': 0.2182939499616623, 'category_id': 1}, {'bbox': [264.6108093261719, 287.2953186035156, 72.81625366210938, 72.44686889648438], 'score': 0.084161676466465, 'category_id': 1}, {'bbox': [273.81103515625, 268.45428466796875, 25.820159912109375, 15.638671875], 'score': 0.17595580220222473, 'category_id': 3}, {'bbox': [264.5993957519531, 265.7889709472656, 31.3447265625, 21.252716064453125], 'score': 0.05825873091816902, 'category_id': 3}, {'bbox': [379.30023193359375, 377.64959716796875, 82.51632690429688, 37.476409912109375], 'score': 0.07632085680961609, 'category_id': 5}, {'bbox': [365.8771667480469, 356.9337463378906, 124.57696533203125, 53.39013671875], 'score': 0.25083476305007935, 'category_id': 5}]

    # predict = [{'bbox': [3.597764253616333, 243.01754760742188, 51.93236041069031, 29.966796875], 'score': 0.17598336935043335, 'category_id': 1}, {'bbox': [144.63735961914062, 347.27386474609375, 97.77436828613281, 57.04205322265625], 'score': 0.09381293505430222, 'category_id': 1}, {'bbox': [136.847412109375, 344.49029541015625, 119.66873168945312, 96.26025390625], 'score': 0.9632590413093567, 'category_id': 1}, {'bbox': [473.7080383300781, 422.9619445800781, 180.59262084960938, 57.692352294921875], 'score': 0.05099703371524811, 'category_id': 1}, {'bbox': [8.857243537902832, 241.3592987060547, 266.48516368865967, 195.85691833496094], 'score': 0.16376793384552002, 'category_id': 1}]
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    image = combine()

    plt.imshow(image)
    currentAxis = plt.gca()
    for i in range(len(predict)):
        if predict[i]['score'] < 0.5:
            continue
        cat = int(predict[i]['category_id'])
        #         print('cat,',cat)
        label_name = labels[cat - 1]
        #         print('label_name,',label_name)
        display_txt = '%s: %.2f' % (label_name, predict[i]['score'])
        pt = predict[i]['bbox']
        coords = (pt[0], pt[1]), pt[2], pt[3]
        color = colors[cat]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    # save_path = os.path.join(img_id + '_inf.png')
    plt.savefig('inf.png', bbox_inches='tight', pad_inches=0.0)
    print('1111111111')


# test()

def test_cm():

    nms = ['void',
           'Front bumper skin',
           'Rear bumper skin',
           'Front bumper grille',
           'network',
           'Hood',
           'Front fog lamp',
           'Headlamp',
           'Front windshield',
           'Roof panel',
           'Front fender',
           'Rear fender',
           'Front door shell outer plate',
           'Rear door shell outer plate',
           'Door glass front',
           'Door glass rear',
           'Rearview mirror assembly',
           'Tires front',
           'Tires rear',
           'Wheel',
           'Luggage cover',
           'Rocker outer plate ',
           'Rear lamp',
           'Rear windshield',
           'External spare tire cover',
           'Wheel trim cover',
           'Front Fender Turn Signal',
           'A pillar',
           'Rear side glass',
           'Front side glass',
           'License plate front',
           'License plate rear']


    def decode_seg_map_sequence(label_masks, dataset='coco'):
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = decode_segmap(label_mask, dataset)
            rgb_masks.append(rgb_mask)
        rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
        return rgb_masks

    def decode_segmap(label_mask, dataset, plot=False):

        if dataset == 'pascal' or dataset == 'coco':
            n_classes = 32
            label_colours = get_pascal_labels()
        else:
            raise NotImplementedError

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def encode_segmap(mask):

        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def get_pascal_labels():

        return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
                           [0, 0, 64], [0, 0, 192], [128, 0, 64], [128, 0, 192],
                           [0, 128, 64], [0, 128, 192], [128, 128, 64], [128, 128, 192]])

    start_time = time.time()

    learn = load_learner('G:\\yzh\\zcm', '512_deeplab_5.pkl')

    # In[6]:

    image = open_image('img.jpg')
    tfm_image = image.apply_tfms(tfms=None, size=512, )  # resize_method=ResizeMethod.PAD,padding_mode='zeros')

    # In[20]:

    pred = learn.predict(tfm_image)

    output = pred[2].unsqueeze(0)
    grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3,
                           normalize=False, range=(0, 255))
    grid_image = grid_image.numpy()
    grid_image = np.moveaxis(grid_image, 0, 2)
    matplotlib.image.imsave('predict.png', grid_image)

    print('ljsdjflkjkldsjkl', time.time() - start_time)


test_cm()