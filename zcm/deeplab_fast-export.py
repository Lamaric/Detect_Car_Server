
# coding: utf-8

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

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
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

from fastai.vision import *
#from modeling.sync_batchnorm.replicate import patch_replication_callback
#from modeling.deeplab import *
import matplotlib, time, os


# In[4]:


#torch.cuda.set_device(1)


# In[5]:
start_time = time.time()
print(os.getcwd())


learn = load_learner('G:\\yzh\\zcm','512_deeplab_5.pkl')


# In[6]:


image = open_image('img.jpg')
tfm_image = image.apply_tfms(tfms=None,size=512,)#resize_method=ResizeMethod.PAD,padding_mode='zeros')


# In[20]:


pred = learn.predict(tfm_image)


# In[21]:


"""record = np.zeros((512,512,3))
for i in range(512):
    for j in range(512):
        record[i,j,:] = rgbList[p[i][j]]
"""
output = pred[2].unsqueeze(0)
grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
grid_image = grid_image.numpy()
grid_image = np.moveaxis(grid_image,0,2)
matplotlib.image.imsave('predict.png', grid_image)
# In[22]:

print('ljsdjflkjkldsjkl', time.time() - start_time)

