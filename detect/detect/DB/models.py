from django.db import models

class User(models.Model):
    name = models.CharField(max_length=32)
    wxid = models.CharField(max_length=32)
    driver = models.CharField(max_length=128)
    driving = models.CharField(max_length=128)


class Recoder(models.Model):
    wxid = models.CharField(max_length=32)
    img = models.CharField(max_length=128)
    result = models.CharField(max_length=128)
    data = models.DateTimeField(auto_now_add=True)