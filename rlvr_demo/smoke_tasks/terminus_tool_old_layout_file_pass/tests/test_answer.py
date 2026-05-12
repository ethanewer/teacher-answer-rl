from pathlib import Path


def test_answer_file_contains_pass():
    assert Path("/app/answer.txt").read_text().strip() == "pass"
