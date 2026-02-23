from src.inference import predict_single

def test_predict_single_returns_expected_keys():
    out = predict_single("This movie was amazing!")
    assert "label" in out
    assert "proba_positive" in out
    assert out["label"] in {"positive", "negative"}
    assert 0.0 <= out["proba_positive"] <= 1.0
