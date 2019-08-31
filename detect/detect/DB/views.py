from DB.models import User, Recoder


def user_write(name, wxid, driver, driving):
    '''
    写 User 表
    :param name: 用户名
    :param wxid: 微信ID
    :param driver: 驾驶证图片路径
    :param driving: 行驶证图片路径
    :return: None
    '''

    user = User()
    user.name = name
    user.wxid = wxid
    user.driver = driver
    user.driving = driving
    user.save()


def record_write(wxid, img_path, result):
    '''
    写record表
    :param wxid: 微信id
    :param img_path: 车损图片路径
    :param result: 检测结果路径
    :return: None
    '''

    record = Recoder()
    record.wxid = wxid
    record.img = img_path
    record.result = result
    record.save()
