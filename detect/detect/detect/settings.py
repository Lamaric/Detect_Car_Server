"""
Django settings for untitled project.

Generated by 'django-admin startproject' using Django 2.2.3.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'vj$xm_7g&^&82cd%+=ro(@^ix4b^s#b&0vqx90z^c4^@zdfz34'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# set ALLOWED_HOSTS
ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'Upload',
    'DB',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'detect.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')]
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'detect.wsgi.application'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
#DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'DETECT',
#         'USER': 'zhihao',
#         'PASSWORD': '123456',
#         'HOST': '192.168.113.128',
#         'POST': '3306',
#     }
# }




# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'
# set time_zone
TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True
# set use_tz
USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
TEST_PATH = r'G:\yzh\detect\detect\detectModel\test1_1.py'
MODEL_PATH = r'G:\yzh\detect\detect\detectModel\checkpoint.pth.tar'
PIC_URL = 'https://lamaric.goho.co/media/pic/'
#MMCV_CONF_DIR = 'G:\\yzh\\mmdir\\mmdir\\mmdetection\\configs\\faster_rcnn_r101_fpn_1x.py'
#MMCV_CKP = 'G:\\yzh\\yzh\\epoch_12.pth'
MMCV_CONF_DIR = 'G:\\yzh\\mmdir\\mmdir\\mmdetection\\configs\\faster_rcnn_r101_fpn_1x_anchor_9_81216.py'
MMCV_CKP = 'G:\\yzh\\yzh\\faster_rcnn_r101_0806_epoch_12.pth'
CM_MODLE_PATH = 'G:\\yzh\\zcm'