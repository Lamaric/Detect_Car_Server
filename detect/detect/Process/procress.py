# -*- coding:utf-8 -*-
"""
@author: Lamarwp
@file:procress.py
@time:2019/8/414:59

"""
from detect.settings import BASE_DIR, TEST_PATH, MODEL_PATH, PIC_URL, MMCV_CONF_DIR, MMCV_CKP, CM_MODLE_PATH
import cv2 as cv
from mmdet.apis import init_detector, inference_detector
from torchvision.utils import make_grid
import matplotlib, time
from fastai.vision import *
from collections import Counter
import itertools
import json

#解决中文显示问题

matplotlib.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

colors = plt.cm.hsv(np.linspace(0, 1, 7)).tolist()
#labels = ['Scrape','Scratch','Deformation', 'Cracking','Damaged','SeriousDamageSevere']

# 损伤程度
labels = ['剐蹭','刮伤','变形', '开裂','破损','严重损伤']
# 配件价格
price_dice = {"前保险杠皮": 100, "后保险杠皮": 200, "前保险杠下格栅": 300, "中网": 400, "发动机盖": 500, "前雾灯": 600,
       "前大灯": 700, "前风挡玻璃": 800, "车顶外板": 900, "前叶子板": 1000, "后叶子板": 1100, "前车门壳外板": 1200,
       "后车门壳外板": 1300, "前车门玻璃": 1400, "后车门玻璃": 1500, "倒车镜总成": 1600, "前轮胎": 1700, "后轮胎": 1800,
       "轮圈": 1900, "行李箱盖": 2000, "门槛外板": 2100, "尾灯": 2200, "后风挡玻璃": 2300, "外挂式备胎罩": 2400, "轮圈装饰罩": 2500,
       "前叶子板转向灯": 2600, "A柱": 2700, "后侧围玻璃": 2800, "前侧围玻璃": 2900, "前牌照板": 3000, "后牌照板": 3100}


def add_layer(source_img, split_img):
    '''
    分割结果与原图叠加
    :param source_img: 原图，------> resize后的原图 （512 * 512）
    :param split_img:  分割后的图片  (512 * 512)
    :return: 叠加后的图片
    '''

    img1 = cv.imread(source_img)
    #img2 = cv.imread(split_img)
    #img2 = img2[: 512, :512, :]

    #dst = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
    # 转成rgb
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

    return img1


def combine(predict, source_img, split_img, inf_path):
    '''
    检测结果与分割结果结合，将结果图保存到inf_path
    :param predict: charming模型输出
    :param source_img: 原图路径
    :param split_img: 分割结果图路径
    :param inf_path: 最终结果图路径
    :return: 置信度大于0.5的 bbox列表
    '''

    bbox_list = [] # bbox列表
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    image = add_layer(source_img, split_img)
    plt.imshow(image)
    currentAxis = plt.gca()

    # 选出置信度大于0.5的bbox
    for i in range(len(predict)):
        if predict[i]['score'] <= 0.5:
            continue
        print(predict[i])
        bbox_list.append(predict[i])
        cat = int(predict[i]['category_id'])
        #         print('cat,',cat)
        label_name = labels[cat - 1]
        #         print('label_name,',label_name)
        display_txt = '%s: %.2f' % (label_name, predict[i]['score'])
        pt = predict[i]['bbox']
        coords = (pt[0], pt[1]), pt[2], pt[3]
        color = colors[cat]
        # 绘制框
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # 添加文本
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    # save_path = os.path.join(img_id + '_inf.png')
    # print('inf_path', inf_path)
    plt.savefig(inf_path, bbox_inches='tight', pad_inches=0.0)

    return bbox_list


def wait_detect_done(path):
    path = os.listdir(path)
    finish_flag = False
    while not finish_flag:
        if len(path) < 2:
            time.sleep(0.2)
        else:
            finish_flag = True


def mmcv_model(config_file, checkpoint_file, img_dir):
    '''
    调用mmcv检测
    :param config_file: 配置文件路径
    :param checkpoint_file: ckp文件
    :param img_dir: 待检图片路径
    :return: 模型输出检测结果， list of bbox, score, category_id
    '''
    model = init_detector(config_file, checkpoint_file)
    result = inference_detector(model, img_dir)
    # show_result(img_dir, result, model.CLASSES,score_thr=0.6, out_file=out_file)
    def xyxy2xywh(bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]
    def det2json(result):
        json_result = []
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = label + 1
                #             print(label)
                json_result.append(data)
        return json_result
    predict = det2json(result)
    return predict


def charming(img_path, split_path, resize_path, model_path):
    '''
    charming 模型
    :param img_path: 原图路径
    :param split_path: 分割结果路径
    :param resize_path: resize 图片路径
    :param model_path: CM_MODEL_PATH
    :return: 模型输出 pred
    '''

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

    def get_pascal_labels():

        return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
                           [0, 0, 64], [0, 0, 192], [128, 0, 64], [128, 0, 192],
                           [0, 128, 64], [0, 128, 192], [128, 128, 64], [128, 128, 192]])

    learn = load_learner(model_path, '512_deeplab_5.pkl')
    image = open_image(img_path)
    # tfm_image = image.apply_tfms(tfms=None,size=512,resize_method=ResizeMethod.PAD,padding_mode='zeros')  # resize_method=ResizeMethod.PAD,padding_mode='zeros')
    tfm_image = image.apply_tfms(tfms=None,size=512)# ,resize_method=ResizeMethod.PAD,padding_mode='zeros')  # resize_method=ResizeMethod.PAD,padding_mode='zeros')
    tfm_image.save(resize_path)
    #print(learn.model.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    learn.model.to(device)
    pred = learn.predict(tfm_image)

    output = pred[2].unsqueeze(0)
    grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3,
                           normalize=False, range=(0, 255))
    grid_image = grid_image.numpy()
    grid_image = np.moveaxis(grid_image, 0, 2)
    matplotlib.image.imsave(split_path, grid_image)

    return pred


def result_output(pred, bbox):
    '''
    输出结果
    :param pred: charming 网络输出
    :param bbox: list of bbox  (score > 0.5)
    :return: 损伤部位列表 damage_part
    '''

    nms = ["背景", "前保险杠皮", "后保险杠皮", "前保险杠下格栅", "中网", "发动机盖", "前雾灯",
           "前大灯", "前风挡玻璃", "车顶外板", "前叶子板", "后叶子板", "前车门壳外板",
           "后车门壳外板", "前车门玻璃", "后车门玻璃", "倒车镜总成", "前轮胎", "后轮胎",
           "轮圈", "行李箱盖", "门槛外板", "尾灯", "后风挡玻璃", "外挂式备胎罩", "轮圈装饰罩",
           "前叶子板转向灯", "A柱", "后侧围玻璃", "前侧围玻璃", "前牌照板", "后牌照板"]
    # print(info)
    damage_part = []
    p = pred[0].data.squeeze()
    for i in bbox:
        bbox = i['bbox']
        int_bbox = list(map(int, bbox))
        seg_cls = p[int_bbox[1]:int_bbox[1] + int_bbox[3], int_bbox[0]:int_bbox[0] + int_bbox[2]]  # slice
        flattened_pred = list(itertools.chain(*seg_cls.numpy()))
        cls = Counter(flattened_pred).most_common(3)  # 计算框中常见类
        all_pixs = len(flattened_pred)
        #print('cls', cls)
        for k, v in cls:
            if k == 0: continue
            if v / all_pixs > 0.3:  # 占比判断
                #print('部件：', nms[k])
                #print(i['category_id'])
                damage_part.append({'part': nms[k], 'type': labels[i['category_id'] - 1]})
                #print(labels[i['category_id'] - 1])

        # print('djokspl', damage_part)

    return damage_part


def split_detect(img_path, split_path, resize_img_path, combine_path):
    '''
    分割和检测
    :param img_path: 原图片路径
    :param split_path: 分割结果路径
    :param resize_img_path: resize图片路径
    :param combine_path: 结果图片路径
    :return: result_stt 损伤结果字符串
    '''

    # hd 模型
    # os.system(
    #     r"python {test_path} --backbone resnet --workers 0 --epochs 10 --batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --resume {model_path} --img_dir {img_path} --save_path {save_path}".format(
    #         test_path = TEST_PATH,
    #         model_path = MODEL_PATH,
    #         img_path = img_path,
    #         save_path = reslt_path
    #     ))
    # wait_detect_done(path)

    # 获取分割模型结果
    pred = charming(img_path, split_path, resize_img_path, CM_MODLE_PATH)
    # 检测模型
    info = mmcv_model(MMCV_CONF_DIR, MMCV_CKP, resize_img_path)
    # 分割检测结合
    bbox_list = combine(info, resize_img_path, split_path, combine_path)
    # 获取损伤部位list
    damage_part = result_output(pred, bbox_list)

    # print(damage_part)

    # list 转为 str
    # result_str = ''
    result_dict = []
    if len(damage_part) >= 1: # 列表不为空
        for i in damage_part:
            # print('i', i)
            result_dict.append({'part': i['part'], 'type': i['type'], 'price': price_dice[i['part']]})
            # result_str += '损伤部位: %s, \n损伤类型: %s , \n维修价格: %s \n' % (i['part'], i['type'], price_dice[i['part']])
            # print('result',result_str)
        for item in result_dict:
             if result_dict.count(item) >1:
                 result_dict.remove(item)

        print(result_dict)

    else:
        # result_str = None
        result_dict = None
    # print(result_str)



    return  result_dict