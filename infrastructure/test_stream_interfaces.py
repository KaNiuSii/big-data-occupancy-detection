import json
import uuid
from datetime import datetime, timezone
import time

from kafka import KafkaProducer, KafkaConsumer

# 🔧 DOSTOSUJ DO ŚRODOWISKA:
# - z hosta: "localhost:9092"
# - z kontenera w sieci docker-compose: "kafka:9092"
KAFKA_BOOTSTRAP_SERVERS = "localhost:29092"

REQUEST_TOPIC = "occupancy_requests"
RESPONSE_TOPIC = "occupancy_responses"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_valid_payload(request_id: str):
    return {
        "request_id": request_id,
        "timestamp": now_iso(),
        "payload": {
            "Temperature": 23.1,
            "Humidity": 27.2,
            "CO2": 704.5,
            "HumidityRatio": 0.00475,
        },
    }


def build_missing_field_payload(request_id: str):
    # Brakuje np. CO2
    return {
        "request_id": request_id,
        "timestamp": now_iso(),
        "payload": {
            "Temperature": 23.1,
            "Humidity": 27.2,
            # "CO2" pominięte
            "HumidityRatio": 0.00475,
        },
    }


def build_null_field_payload(request_id: str):
    # CO2 ustawione na null
    return {
        "request_id": request_id,
        "timestamp": now_iso(),
        "payload": {
            "Temperature": 23.1,
            "Humidity": 27.2,
            "CO2": None,
            "HumidityRatio": 0.00475,
        },
    }


def build_wrong_type_payload(request_id: str):
    # Temperature jako string zamiast liczby
    return {
        "request_id": request_id,
        "timestamp": now_iso(),
        "payload": {
            "Temperature": "23.1",   # zły typ
            "Humidity": 27.2,
            "CO2": 704.5,
            "HumidityRatio": 0.00475,
        },
    }


def build_broken_payload(request_id: str):
    # Całkiem niezgodne z oczekiwanym schematem
    return {
        "request_id": request_id,
        "timestamp": now_iso(),
        "payload": {
            "foo": "bar",
            "something": 123,
        },
    }


def main():
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
    )

    consumer = KafkaConsumer(
        RESPONSE_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        auto_offset_reset="earliest",  # tylko nowe wiadomości
        enable_auto_commit=True,
        group_id=f"occupancy-test-client-{uuid.uuid4()}",
        value_deserializer=lambda m: m.decode("utf-8"),
    )

    # Przygotowujemy różne przypadki testowe
    tests = [
        ("valid", build_valid_payload),
        ("missing_field", build_missing_field_payload),
        ("null_field", build_null_field_payload),
        ("wrong_type", build_wrong_type_payload),
        ("broken_payload", build_broken_payload),
    ]

    # Mapujemy request_id -> label testu (żeby łatwiej rozpoznać w odpowiedziach)
    request_id_to_label = {}

    print("=== Wysyłam testowe wiadomości na topic occupancy_requests ===")
    for label, builder in tests:
        rid = str(uuid.uuid4())
        payload = builder(rid)
        request_id_to_label[rid] = label

        producer.send(
            REQUEST_TOPIC,
            key=rid,   # niekonieczne, ale spoko
            value=payload,
        )
        print(f"[SENT] {label} | request_id={rid} | payload={payload}")

    producer.flush()
    print("=== Wszystkie requesty wysłane, czekam na odpowiedzi... ===")

    # Czekamy na odpowiedzi – zbieramy, aż dostaniemy wszystkie albo minie timeout
    expected_count = len(request_id_to_label)
    received = {}

    timeout_seconds = 20
    start = time.time()

    while len(received) < expected_count and (time.time() - start) < timeout_seconds:
        msg_pack = consumer.poll(timeout_ms=500)
        for tp, messages in msg_pack.items():
            for msg in messages:
                try:
                    value_str = msg.value
                    data = json.loads(value_str)
                except Exception as e:
                    print(f"[WARN] Nie udało się sparsować odpowiedzi: {msg.value} ({e})")
                    continue

                rid = data.get("request_id")
                if rid in request_id_to_label:
                    label = request_id_to_label[rid]
                    received[rid] = (label, data)
                    print(f"[RECV] {label} | request_id={rid} | data={data}")

    print("\n=== PODSUMOWANIE TESTU ===")
    for rid, label in request_id_to_label.items():
        if rid in received:
            _, data = received[rid]
            print(f"[OK] {label} → prediction={data.get('prediction')}, probability={data.get('probability')}")
        else:
            print(f"[MISSING] Brak odpowiedzi dla {label} (request_id={rid})")

    print("\nKoniec testu.")


if __name__ == "__main__":
    main()
