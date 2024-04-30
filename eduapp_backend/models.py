from django.db import models

class Video(models.Model):
    url = models.URLField()
    thumbnail_url = models.URLField()
    notes = models.TextField()