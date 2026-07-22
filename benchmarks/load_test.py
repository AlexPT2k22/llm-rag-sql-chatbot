"""
Teste de carga concorrente
Simula múltiplos utilizadores a fazer perguntas ao chatbot em simultâneo.

Execução headless (sem browser, gera CSV):
    locust -f load_test.py --host http://localhost:8000 --headless -u 5 -r 1 --run-time 2m --csv load_test_stats
"""

import random
from locust import HttpUser, task, between

QUESTIONS_RAG = [
    "O que é o AgriSystem?",
    "Como registo uma nova parcela?",
    "O que são índices de vegetação NDVI?",
    "Como faço a gestão de fitofármacos?",
    "Como consulto o histórico de práticas agrícolas?",
    "O que é a monitorização de pragas?",
    "Como adiciono um equipamento ao sistema?",
    "O que é o módulo de cellar?",
    "Como registo uma fermentação?",
    "O que são as castas de vinhas velhas?",
]

QUESTIONS_SQL = [
    "Quantas parcelas existem no sistema?",
    "Quais são os tipos de solo disponíveis?",
    "Quantos fitofármacos estão registados?",
    "Quais são os distritos disponíveis no sistema?",
]

class ChatUser(HttpUser):
    """Utilizador que usa o endpoint de streaming SSE."""
    wait_time = between(2, 5)

    def on_start(self):
        self.session_id = f"load_{random.randint(10000, 99999)}"

    @task(3)
    def ask_rag(self):
        self._ask(random.choice(QUESTIONS_RAG))

    @task(1)
    def ask_sql(self):
        self._ask(random.choice(QUESTIONS_SQL))

    def _ask(self, question: str):
        params = {
            "message": question, 
            "session_id": self.session_id,
            "entidade_id": 8,
            "ano_agricola_id": 32
        }
        
        with self.client.get(
            "/chat/stream",
            params=params,
            catch_response=True,
            name="/chat/stream",
            timeout=120 
        ) as resp:
            
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif not resp.text.strip():
                resp.failure("A resposta chegou completamente vazia.")
            else:
                resp.success()