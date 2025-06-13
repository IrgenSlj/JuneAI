from django.db import models
from django.utils import timezone

class Conversation(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=200, default="New Chat")

    def __str__(self):
        return f"Chat {self.id} - {self.title}"

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    content = models.TextField()
    role = models.CharField(max_length=50)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."