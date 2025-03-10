from django.db import models
from django.utils import timezone

class Conversation(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=200, default="New Chat")

    class Meta:
        app_label = 'chat'
        db_table = 'chat_conversation'  # Explicitly set the table name

    def __str__(self):
        return f"Chat {self.id} - {self.title}"

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    content = models.TextField()
    role = models.CharField(max_length=50)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['timestamp']
        app_label = 'chat'
        db_table = 'chat_message'  # Explicitly set the table name

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
