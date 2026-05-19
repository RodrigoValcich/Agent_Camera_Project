"""
Assistente IA usando OpenAI GPT-4o (com suporte a visão).

Dependência: pip install openai
Chave:       OPENAI_API_KEY no .env
"""
import base64
import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai não instalado — rode: pip install openai")

SYSTEM_PROMPT = """Você é um assistente de segurança residencial inteligente.
Você monitora câmeras de uma casa e responde perguntas sobre o que acontece nelas.

Ao analisar imagens de câmeras:
- Descreva o que vê de forma clara e objetiva
- Identifique pessoas, veículos, animais e atividades suspeitas
- Informe horários e câmeras quando relevante
- Se houver algo preocupante, destaque imediatamente
- Seja conciso mas completo

Responda sempre em português do Brasil."""


def _encode_frame(image: np.ndarray, quality: int = 85) -> str:
    """Converte frame OpenCV (BGR) para base64 JPEG."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _file_to_b64(path: str) -> Optional[str]:
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return _encode_frame(img)
    except Exception as e:
        logger.error(f"Erro ao carregar imagem {path}: {e}")
        return None


class AIAssistant:
    """Assistente IA usando OpenAI GPT-4o com suporte a visão."""

    def __init__(self, config: dict):
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai não instalado.\n"
                "Execute: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não encontrada no .env")

        self._client    = AsyncOpenAI(api_key=api_key)
        ai_cfg          = config.get("ai", {})
        self.model      = ai_cfg.get("model", "gpt-4o-mini")  # gpt-4o-mini é mais barato
        self.max_tokens = ai_cfg.get("max_tokens", 1024)
        self.max_frames = ai_cfg.get("max_frames_per_query", 3)

        # Histórico do chat
        self._history: list[dict] = []

        logger.info(f"OpenAI AIAssistant iniciado: {self.model} ✓")

    # ------------------------------------------------------------------ #

    async def analyze_event(self, event, thumbnail_path: str) -> str:
        """Gera resumo automático de um evento detectado."""
        b64 = _file_to_b64(thumbnail_path)
        if not b64:
            return f"Evento detectado: {event.event_type} em {event.camera_name}"

        detected = ", ".join(set(d.label for d in event.detections)) if event.detections else event.event_type
        ts = datetime.fromtimestamp(event.timestamp).strftime("%d/%m/%Y às %H:%M:%S")

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                        {"type": "text", "text": (
                            f"Analise esta imagem da câmera '{event.camera_name}' capturada {ts}. "
                            f"O sistema detectou: {detected}. "
                            "Descreva em 1-2 frases o que está acontecendo."
                        )},
                    ],
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erro na análise de evento: {e}")
            return f"{detected} detectado em {event.camera_name} às {ts}."

    async def chat(
        self,
        user_message: str,
        frames: Optional[list] = None,
        event_context: Optional[dict] = None,
    ) -> str:
        """Chat com histórico. Aceita frames ao vivo e/ou contexto de evento."""
        content: list[dict] = []

        # Contexto de evento
        if event_context:
            content.append({"type": "text", "text": (
                f"[Contexto: evento '{event_context.get('event_type')}' "
                f"na câmera '{event_context.get('camera_name')}' "
                f"às {event_context.get('time', '')}. "
                f"Resumo: {event_context.get('ai_summary', '')}]"
            )})

        # Frames de câmera
        if frames:
            for img in frames[-self.max_frames:]:
                b64 = _encode_frame(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })

        content.append({"type": "text", "text": user_message})

        self._history.append({"role": "user", "content": content})

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *self._history,
                ],
            )
            reply = response.choices[0].message.content.strip()
            self._history.append({"role": "assistant", "content": reply})

            # Limita histórico
            if len(self._history) > 40:
                self._history = self._history[-40:]

            return reply

        except Exception as e:
            logger.error(f"Erro no chat GPT: {e}")
            return f"Desculpe, ocorreu um erro: {e}"

    async def describe_live(self, frame: np.ndarray, camera_name: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        return await self.chat(
            f"O que está acontecendo agora na câmera '{camera_name}'? (horário: {ts})",
            frames=[frame],
        )

    def clear_history(self):
        self._history.clear()