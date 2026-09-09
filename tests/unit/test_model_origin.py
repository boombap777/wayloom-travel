import hashlib

from travel_itinerary.model_origin import recorded_origin_matches, verify_directory


def test_origin_depends_on_weights_not_the_historical_base_label(tmp_path):
    payload = b"test weights"
    row = {
        "filename": "model.safetensors",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    origin = {"model_id": "Qwen/Qwen3-1.7B", "revision": "test", "weights": [row]}
    assert recorded_origin_matches([row], origin)
    assert not recorded_origin_matches([{**row, "sha256": "wrong"}], origin)
    assert not verify_directory(tmp_path, origin)["passed"]
    (tmp_path / row["filename"]).write_bytes(payload)
    assert verify_directory(tmp_path, origin)["passed"]
