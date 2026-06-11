from src.components.agents.graph import graph

sample_transaction = {
    "transaction_id": "TXN100001",
    "customer_id": "CUST001",
    "beneficiary_id": "BEN001",
    "merchant_id": "MER001",
    "device_id": "DEV001",
    "transaction_timestamp": "2026-06-11T10:00:00Z",
    "transaction_type": "TRANSFER",
    "transaction_amount": 250000,
    "currency": "INR",
    "payment_method": "UPI",
    "ip_address": "103.22.10.5",
    "origin_country": "IN",
    "destination_country": "SG",
    "transaction_status": "SUCCESS",
    "is_international": True,
    "features_for_classifier": {
        "account_age_days": 30,
        "transaction_frequency_24h": 35,
        "failed_transaction_count_24h": 4,
        "avg_transaction_amount_7d": 15000,
        "session_duration_minutes": 2,
        "device_risk_score": 92.5,
        "unusual_amount_flag": True,
        "unusual_location_flag": True,
        "typing_speed_flag": True,
    },
    "agent_pipelines_telemetry": {
        "initial_llm_probability": 0.87,
        "initial_risk_category": "HIGH",
        "orchestrator_decision": "INVESTIGATE",
        "behavioral_agent_context": {
            "biometric_anomaly_detected": True,
            "automation_script_suspected": True,
        },
        "graph_agent_context": {
            "shared_device_mule_count": 8,
            "known_fraud_ring_edge": True,
        },
        "risk_agent_context": {
            "sanction_list_match": False,
            "pep_flag": False,
            "beneficiary_risk_rating": "HIGH",
        },
    },
}


def main():

    state = {
        "transaction": sample_transaction,
        "iteration_count": 0,
        "confidence_score": 0,
        "messages": [],
    }

    result = graph.invoke(state)

    print(result)


if __name__ == "__main__":
    main()
