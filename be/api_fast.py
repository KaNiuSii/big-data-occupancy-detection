import os
import json
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kafka import KafkaProducer, KafkaConsumer
import uvicorn

# === LOGOWANIE ===
logger = logging.getLogger("occupancy_api")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

# === Konfiguracja Kafki ===
# Backend odpalasz na hoście, więc korzystamy z PLAINTEXT_EXTERNAL://localhost:29092
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
REQUEST_TOPIC = "occupancy_requests"
RESPONSE_TOPIC = "occupancy_responses"
RESPONSE_TIMEOUT_SECONDS = 5.0

app = FastAPI(title="Occupancy Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

producer: KafkaProducer | None = None


# --- MODELE Pydantic (kontrakt API) ---


class Features(BaseModel):
    Temperature: Optional[float] = Field(None, description="Celsius")
    Humidity: Optional[float] = None
    CO2: Optional[float] = None
    HumidityRatio: Optional[float] = None


class PredictRequest(BaseModel):
    payload: Features
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class PredictionResult(BaseModel):
    request_id: str
    timestamp: Optional[datetime]
    features: Dict[str, Any]
    prediction: int
    probability: float


def create_producer() -> KafkaProducer:
    logger.info("[KAFKA] Tworzę KafkaProducer na %s", KAFKA_BOOTSTRAP_SERVERS)
    return KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
    )


@app.on_event("startup")
def on_startup():
    global producer
    producer = create_producer()
    logger.info("[API] Startup zakończony, połączono z Kafka @ %s", KAFKA_BOOTSTRAP_SERVERS)


@app.on_event("shutdown")
def on_shutdown():
    global producer
    logger.info("[API] Shutdown – zamykam KafkaProducer")
    if producer is not None:
        producer.flush()
        producer.close()
        producer = None


@app.post("/predict", response_model=PredictionResult)
def predict(req: PredictRequest):
    """
    Przyjmuje dane czujników, wysyła je na Kafkę,
    czeka na odpowiedź ze Spark Streaming i zwraca wynik.
    """
    if producer is None:
        logger.error("[/predict] Kafka producer not initialized")
        raise HTTPException(status_code=500, detail="Kafka producer not initialized")

    # 1. Przygotuj ID i timestamp
    request_id = req.request_id or str(uuid.uuid4())
    ts = req.timestamp or datetime.now(timezone.utc)

    payload_dict = req.payload.model_dump(exclude_unset=True)

    logger.info(
        "[/predict] NOWE ŻĄDANIE: request_id=%s, timestamp=%s, payload=%s",
        request_id,
        ts.isoformat(),
        payload_dict,
    )

    kafka_message = {
        "request_id": request_id,
        "timestamp": ts.isoformat(),
        "payload": payload_dict,
    }

    # --- Konsument do nasłuchiwania odpowiedzi ---
    logger.info("[/predict] Tworzę KafkaConsumer na topic=%s", RESPONSE_TOPIC)
    consumer = KafkaConsumer(
        RESPONSE_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id=None,  # anonimowy consumer
        value_deserializer=lambda m: m.decode("utf-8"),
    )

    # Teraz wysyłamy request
    logger.info(
        "[KAFKA→REQUEST] Wysyłam na topic=%s: key=%s, value=%s",
        REQUEST_TOPIC,
        request_id,
        kafka_message,
    )

    producer.send(
        REQUEST_TOPIC,
        key=request_id,
        value=kafka_message,
    )
    producer.flush()

    deadline = time.time() + RESPONSE_TIMEOUT_SECONDS
    found = None

    try:
        while time.time() < deadline and found is None:
            msg_pack = consumer.poll(timeout_ms=500)
            if not msg_pack:
                logger.debug("[/predict] poll() – brak wiadomości, czekam dalej...")
            for tp, messages in msg_pack.items():
                for msg in messages:
                    try:
                        data = json.loads(msg.value)
                    except Exception as e:
                        logger.warning(
                            "[KAFKA←RESPONSE] Nie udało się zdekodować JSON z value=%s, error=%s",
                            msg.value,
                            e,
                        )
                        continue

                    logger.info(
                        "[KAFKA←RESPONSE] Otrzymano wiadomość z topic=%s partition=%s offset=%s: %s",
                        tp.topic,
                        tp.partition,
                        msg.offset,
                        data,
                    )

                    if data.get("request_id") == request_id:
                        logger.info(
                            "[KAFKA←RESPONSE] Dopasowano response dla request_id=%s",
                            request_id,
                        )
                        found = data
                        break
                if found is not None:
                    break
    finally:
        consumer.close()
        logger.info("[/predict] Zamknięto consumer")

    if found is None:
        logger.warning(
            "[/predict] TIMEOUT – brak odpowiedzi dla request_id=%s w czasie %ss",
            request_id,
            RESPONSE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=504,
            detail="Timeout waiting for prediction from streaming engine",
        )

    # Odpowiedź Sparka ma JSON:
    # { request_id, timestamp, features, prediction, probability }
    ts_resp = found.get("timestamp")
    ts_parsed = None
    if ts_resp:
        try:
            ts_parsed = datetime.fromisoformat(ts_resp)
        except Exception as e:
            logger.warning(
                "[/predict] Nie udało się sparsować timestamp=%s, error=%s",
                ts_resp,
                e,
            )
            ts_parsed = None

    pred = int(found.get("prediction", -1))
    prob = float(found.get("probability", -1.0))
    feats = found.get("features", {})

    logger.info(
        "[/predict] GOTOWA ODPOWIEDŹ: request_id=%s, prediction=%s, probability=%s, features=%s",
        request_id,
        pred,
        prob,
        feats,
    )

    return PredictionResult(
        request_id=found.get("request_id", request_id),
        timestamp=ts_parsed,
        features=feats,
        prediction=pred,
        probability=prob,
    )


if __name__ == "__main__":
    # Dzięki temu odpalasz po prostu:
    #   python api_fast.py
    logger.info("[API] Uruchamiam Uvicorn na 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
