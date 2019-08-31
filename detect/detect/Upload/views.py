from django.shortcuts import render
from django.http import HttpResponse
from detect.settings import BASE_DIR, TEST_PATH, MODEL_PATH, PIC_URL, MMCV_CONF_DIR, MMCV_CKP
from Process import procress
import time
from fastai.vision import *



def upload(request):
    '''
    post方法上传车损图片，返回检测图片的URL和result
    :param request:
    :return:
    '''
    if request.method == 'POST':  # 获取对象
        obj = request.FILES.get('file')

        # 以时间戳命名图片文件及文件夹， 并储存在 detect/media/pic下
        img_name = str(int(time.time() * 1000)) + '.jpg'
        # 文件夹名
        path = BASE_DIR + '\\media\\pic\\%s\\' % img_name.split('.')[0]
        # 创建文件夹
        os.mkdir(path)
        # 在创建的文件夹下存原图片
        img_path = path + img_name
        split_path = path + 'split_result.jpg'
        resize_img_path = path + 'resize.jpg'
        combine_path = path + 'inf.jpg'
        with open(img_path, 'wb') as f:
            for chunk in obj.chunks():
                f.write(chunk)

        # 调用模型，返回检测结果
        start_time = time.time()
        # result_str = procress.split_detect(img_path, split_path, resize_img_path, combine_path)
        # print(result_str)
        # print('time', time.time() - start_time)
        result_dict = procress.split_detect(img_path, split_path, resize_img_path, combine_path)
        print(result_dict)
        print('time', time.time() - start_time)

        return HttpResponse(json.dumps({'result': result_dict,
                                        # 'result_url': PIC_URL + img_name.split('.')[0] + r'/split_result.jpg',
                                        # 'detect_url': PIC_URL + img_name.split('.')[0] + r'/detect_result.jpg',
                                        'combine_url': PIC_URL + img_name.split('.')[0] + r'/inf.jpg'
                                        }))
    return render(request, 'upload.html')