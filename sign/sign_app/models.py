from django.db import models

class SignHistory(models.Model):
    text = models.CharField(max_length=255)
    video_file = models.CharField(max_length=255)
    lang = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.text} ({self.lang}) - {self.created_at}"