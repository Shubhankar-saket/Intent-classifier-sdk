from pathlib import Path

from conftest import ROOT, llm_answer

from intent_classify.backends import HashingEmbedder
from intent_classify.evals import check_gates, load_cases, render, run_eval, validate_against


def test_all_datasets_load_and_match_taxonomy(taxonomy):
    cases = load_cases(sorted((ROOT / "evals" / "datasets").glob("*.jsonl")))
    assert len(cases) > 150
    assert validate_against(cases, taxonomy) == []


def test_run_eval_end_to_end(taxonomy, fake_llm, tmp_path: Path):
    cases = load_cases([ROOT / "evals" / "datasets" / "shopease_multiturn.jsonl"])[:6]

    def respond(system, user):
        if "Classify the user's message into exactly one" in system:  # zero-shot baseline
            return {"label": "affirm"}
        return llm_answer([("order.cancel", 0.9)])

    run = run_eval(
        taxonomy, cases, ["zero_shot", "embeddings", "full"],
        lambda: fake_llm(respond), HashingEmbedder(), out_dir=tmp_path, progress=lambda _: None,
    )
    full = run["systems"]["full"]["summary"]
    assert full["n_cases"] == 6
    assert full["headline"]["context_dependent_accuracy"][0] is not None
    m001 = next(r for r in run["systems"]["full"]["rows"] if r["id"] == "m-001")
    assert m001["intent_correct"] is True
    zs = next(r for r in run["systems"]["zero_shot"]["rows"] if r["id"] == "m-001")
    assert zs["intent_correct"] is False  # no history -> can't resolve "yes"
    html = render(run)
    assert "<table" in html and "full" in html

    gates = tmp_path / "gates.yaml"
    gates.write_text("system: full\nmin: {context_dependent_accuracy: 0.0}\nmax: {benign_false_block_rate: 1.0}\n")
    assert check_gates(run["path"], gates) == []
    gates.write_text("system: full\nmin: {intent_accuracy: 1.01}\n")
    assert check_gates(run["path"], gates)
