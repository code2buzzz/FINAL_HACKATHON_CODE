import os
import sys
import time
import json
import random
import uuid
from datetime import datetime
import numpy as np
from confluent_kafka import Producer
from dotenv import load_dotenv
from config.settings import KAFKA_CONFIG

# Load environment configuration
load_dotenv("")


config = {
    "bootstrap.servers": KAFKA_CONFIG["bootstrap.servers"],
    "client.id": KAFKA_CONFIG["client.id"],
}
producer = Producer(config)
TOPIC_NAME = "fraud-transactions"


def delivery_report(err, msg):
    if err is not None:
        print(f"[-] Message delivery failed: {err}")
    else:
        print(
            f"[+] Broadcasted Event -> Partition [{msg.partition()}] at offset {msg.offset()}"
        )


# --- 1. SEEDING SCHEMA-ALIGNED SIMULATION DATA ---
CUSTOMERS = [f"CUST_{10000 + i}" for i in range(500)]
FRAUD_RING_CUSTOMERS = CUSTOMERS[:40]  # Mule accounts

DEVICES = [f"DEV_{i:05d}" for i in range(100)]
HIGH_RISK_DEVICES = DEVICES[:10]  # Tainted hardware assets

MERCHANTS = [f"MER_{i:03d}" for i in range(30)]
HIGH_RISK_MERCHANTS = MERCHANTS[:3]

BENEFICIARIES = [f"BEN__{20000 + i}" for i in range(200)]
HIGH_RISK_BENEFICIARIES = BENEFICIARIES[:15]  # Flagged receiver accounts

COUNTRIES = ["IN", "US", "AE", "GB", "SG", "MY", "CH"]
RISK_CATEGORIES = ["LOW", "MEDIUM", "HIGH"]
PAYMENT_METHODS = ["UPI", "CREDIT_CARD", "NET_BANKING", "WALLET"]
TRANSACTION_STATUSES = [
    "SUCCESS",
    "FAILED",
    "PENDING",
    "REVERSED",
]

print("[*] Initialized Advanced Relational Models. Pumping live events to Kafka...")

try:
    while True:
        # --- 2. RELATIONAL ENTITY SELECTION ---
        customer = random.choice(CUSTOMERS)
        device = random.choice(DEVICES)
        merchant = random.choice(MERCHANTS)
        beneficiary = random.choice(BENEFICIARIES)
        tx_id = f"TXN_{uuid.uuid4().hex[:12].upper()}"

        # --- 3. INJECTED FRAUD HEURISTIC LOGIC ---
        is_fraud = False
        if customer in FRAUD_RING_CUSTOMERS:
            is_fraud = True
        elif device in HIGH_RISK_DEVICES and random.random() < 0.6:
            is_fraud = True
        elif merchant in HIGH_RISK_MERCHANTS and random.random() < 0.5:
            is_fraud = True
        elif beneficiary in HIGH_RISK_BENEFICIARIES and random.random() < 0.7:
            is_fraud = True

        # --- 4. DYNAMIC LOGNORMAL FINANCIAL ENGINE ---
        if is_fraud:
            amount = round(float(np.random.lognormal(9.5, 0.5)), 2)  # Higher values
            risk_cat = "HIGH"
            fraud_prob = round(random.uniform(75.0, 99.9), 2)
            decision = random.choice(["BLOCKED", "REVIEW_REQUIRED"])

            if decision == "BLOCKED":
                transaction_status = "FAILED"
            else:
                transaction_status = random.choice(["PENDING", "FAILED", "REVERSED"])

        else:
            amount = round(
                float(np.random.lognormal(6.5, 0.7)), 2
            )  # Lower, safer values
            risk_cat = "LOW" if random.random() < 0.85 else "MEDIUM"
            fraud_prob = round(random.uniform(0.1, 45.0), 2)
            decision = "APPROVED" if risk_cat == "LOW" else "REVIEW_REQUIRED"
            if decision == "APPROVED":
                transaction_status = "SUCCESS"
            else:
                transaction_status = random.choice(["PENDING", "SUCCESS"])

        # --- 5. NETWORK & TELEMETRY ANOMALIES ---
        origin_country = "IN" if not is_fraud else random.choice(COUNTRIES)
        # 30% chance of cross-border mismatch if fraudulent
        dest_country = (
            random.choice(COUNTRIES) if (is_fraud and random.random() < 0.3) else "IN"
        )
        ip_address = (
            f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
            if not is_fraud
            else f"{random.randint(40,220)}.{random.randint(10,250)}.{random.randint(0,255)}.{random.randint(1,254)}"
        )

        # --- 6. SYNTHESIZING COMPREHENSIVE STREAM PAYLOAD ---
        event_payload = {
            "transaction_id": tx_id,
            "customer_id": customer,
            "beneficiary_id": beneficiary,
            "merchant_id": merchant,
            "device_id": device,
            "transaction_timestamp": datetime.utcnow().isoformat() + "Z",
            "transaction_type": random.choice(["TRANSFER", "PAYMENT", "WITHDRAWAL"]),
            "transaction_amount": amount,
            "currency": "INR",
            "payment_method": random.choice(PAYMENT_METHODS),
            "ip_address": ip_address,
            "origin_country": origin_country,
            "destination_country": dest_country,
            "transaction_status": transaction_status,
            "is_international": origin_country != dest_country,
            # Data explicitly formatted to feed downstream LLMs / SHAP
            "features_for_classifier": {
                "account_age_days": random.randint(10, 1500),
                "transaction_frequency_24h": (
                    random.randint(15, 60) if is_fraud else random.randint(1, 5)
                ),
                "failed_transaction_count_24h": (
                    random.randint(3, 10) if (is_fraud and random.random() < 0.4) else 0
                ),
                "avg_transaction_amount_7d": round(
                    float(np.random.lognormal(7.0, 0.5)), 2
                ),
                "session_duration_minutes": random.randint(1, 45),
                "device_risk_score": (
                    round(random.uniform(70.0, 100.0), 2)
                    if device in HIGH_RISK_DEVICES
                    else round(random.uniform(0.0, 35.0), 2)
                ),
                "unusual_amount_flag": is_fraud,
                "unusual_location_flag": origin_country != "IN",
                "typing_speed_flag": is_fraud and random.random() < 0.6,
            },
            # Core data inputs engineered for LangGraph Agents
            "agent_pipelines_telemetry": {
                "initial_llm_probability": fraud_prob,
                "initial_risk_category": risk_cat,
                "orchestrator_decision": decision,
                "behavioral_agent_context": {
                    "biometric_anomaly_detected": is_fraud,
                    "automation_script_suspected": is_fraud and random.random() < 0.8,
                },
                "graph_agent_context": {
                    "shared_device_mule_count": (
                        random.randint(4, 15) if is_fraud else 1
                    ),
                    "known_fraud_ring_edge": is_fraud,
                },
                "risk_agent_context": {
                    "sanction_list_match": is_fraud and random.random() < 0.15,
                    "pep_flag": random.random() < 0.02,  # Politically Exposed Person
                    "beneficiary_risk_rating": (
                        "HIGH" if beneficiary in HIGH_RISK_BENEFICIARIES else "LOW"
                    ),
                },
            },
        }

        # --- 7. EMIT TO KAFKA BROKER ---
        producer.produce(
            topic=TOPIC_NAME,
            key=customer,
            value=json.dumps(event_payload).encode("utf-8"),
            callback=delivery_report,
        )

        producer.poll(0)
        time.sleep(1.0)  # Emits 1 transaction per second

except KeyboardInterrupt:
    print("\n[*] Gracefully stopping the stream producer engine.")
finally:
    producer.flush()
