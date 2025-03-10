from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import requests
import json
from .models import Conversation, Message
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import time
import psutil
import logging
from django.conf import settings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self):
        self.port = getattr(settings, 'OLLAMA_PORT', 11434)
        self.host = getattr(settings, 'OLLAMA_HOST', 'localhost')
        self.url = getattr(settings, 'OLLAMA_URL', f'http://{self.host}:{self.port}/api/chat')

class OllamaConnection:
    @staticmethod
    def is_port_in_use(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except Exception as e:
            logger.error(f"Port check error: {e}")
            return False

    @staticmethod
    def kill_process_on_port(port):
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.connections():
                        if conn.laddr.port == port:
                            proc.kill()
                            time.sleep(1)
                            logger.info(f"Killed process on port {port}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Process access error: {e}")
            return False
        except Exception as e:
            logger.error(f"Process kill error: {e}")
            return False

    @staticmethod
    def wait_for_port(port, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if OllamaConnection.is_port_in_use(port):
                return True
            time.sleep(1)
        return False

class JuneAI:
    def __init__(self):
        self.url = 'http://localhost:11434/api/chat'
        self.session = None
        self._ensure_ollama_available()
        self.session = self._create_session()

    def _ensure_ollama_available(self):
        if not OllamaConnection.is_port_in_use(11434):
            logger.warning("Ollama not detected. Please start Ollama service.")
            if not OllamaConnection.wait_for_port(11434):
                raise RuntimeError("Ollama service not available. Run 'ollama serve' first.")

    def _create_session(self):
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('http://', adapter)
        return session

    def get_response(self, conversation_id, message):
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            history = Message.objects.filter(conversation=conversation)
            
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in history
            ]
            
            response = self.session.post(
                self.url,
                json={
                    "model": "june",
                    "messages": messages + [{"role": "user", "content": message}]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            
            logger.error(f"Ollama error: {response.status_code}")
            return "Sorry, I'm having trouble responding right now."
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return "Error: Cannot connect to Ollama. Please restart the service."
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"

june = JuneAI()

@ensure_csrf_cookie
def chat_view(request):
    try:
        conversations = Conversation.objects.all().order_by('-created_at')
        return render(request, 'chat/chat.html', {'conversations': conversations})
    except Exception as e:
        logger.error(f"Chat view error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def send_message(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
        
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        if not conversation_id:
            conversation = Conversation.objects.create()
            conversation_id = conversation.id
            logger.info(f"Created new conversation: {conversation_id}")
        
        # Save user message
        Message.objects.create(
            conversation_id=conversation_id,
            content=message,
            role='user'
        )
        
        # Get June's response
        june_response = june.get_response(conversation_id, message)
        
        # Save June's response
        Message.objects.create(
            conversation_id=conversation_id,
            content=june_response,
            role='assistant'
        )
        
        return JsonResponse({
            'response': june_response,
            'conversation_id': conversation_id
        })
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        return JsonResponse({'error': str(e)}, status=500)