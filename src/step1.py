import json
import math
import re
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "unsloth/Qwen3-4B"
N_SAMPLES = 10            # answers per question
TARGET_QUESTIONS = 500    # 250 train + 250 test
TOPK = 10
BATCH_QUESTIONS = 128     # how many questions to send to vLLM at once
OUT_PATH = "answers.jsonl"

TRAIN_CSV = "gsm8k_tr_train_data_250_sample.csv"
TEST_CSV = "gsm8k_tr_test_data_250_sample.csv"

NUM_RE = re.compile(r"-?\d[\d.,]*")

def normalize_number(s: str):
    """'1.234,56' / '1,234.56' / '42' -> float, best effort."""
    s = str(s).strip().rstrip(".")
    if "," in s and "." in s:
        # decide which is the decimal separator by position
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # treat a single trailing ',xx' as decimal, otherwise thousands sep
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = parts[0] + "." + parts[1]
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def predicted_answer(text: str):
    """Prefer '#### x'; otherwise fall back to the last number in the text."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = NUM_RE.findall(tail)
        if m:
            return normalize_number(m[0])
    m = NUM_RE.findall(text)
    return normalize_number(m[-1]) if m else None

def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return math.isclose(pred, gold, rel_tol=1e-4, abs_tol=1e-6)
    except Exception:
        return False

SYSTEM = (
    "Sen yardımcı bir matematik asistanısın. Problemi adım adım çöz ve "
    "son cevabı mutlaka '#### <sayı>' formatında, tek satırda ver."
)

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def load_local_dataframe():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    df_train["split"] = "train"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096)

    sampling = SamplingParams(
        n=N_SAMPLES,
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
        logprobs=TOPK,       
    )

    df = load_local_dataframe()
    df_train = df[df["split"] == "train"].reset_index(drop=True).iloc[:250]
    df_test = df[df["split"] == "test"].reset_index(drop=True).iloc[:250]
    df = pd.concat([df_train, df_test], ignore_index=True)

    total_questions = len(df)
    assert total_questions == 500, f"Expected 500 questions, got {total_questions}"

    mixed_found = 0
    used_question_ids = set()
    out_f = open(OUT_PATH, "w", encoding="utf-8")
    mixed_bar = tqdm(total=TARGET_QUESTIONS, desc="Mixed questions found",
                     unit="q", position=0)
    scan_bar = tqdm(total=total_questions, desc="Dataset questions scanned",
                    unit="q", position=1)

    for start in range(0, total_questions, BATCH_QUESTIONS):
        if mixed_found >= TARGET_QUESTIONS:
            break
        end = min(start + BATCH_QUESTIONS, total_questions)
        batch = df.iloc[start:end]
        prompts = [build_prompt(tokenizer, q) for q in batch["question"]]
        golds = list(batch["result_math"])

        results = llm.generate(prompts, sampling)
        scan_bar.update(len(batch))

        for qi, req in enumerate(results):
            current_question_id = batch.index[qi]
            if mixed_found >= TARGET_QUESTIONS:
                break
            if current_question_id in used_question_ids:
                continue

            gold = golds[qi]
            try:
                gold = float(gold)
            except Exception:
                continue
            if pd.isnull(gold):
                continue

            records, labels = [], []
            for si, comp in enumerate(req.outputs):
                # per-token sum of the 10 highest probabilities
                top10_sums = []
                for pos_logprobs in comp.logprobs:
                    # vLLM may return topk (+ sampled token) -> keep 10 largest
                    lps = sorted((lp.logprob for lp in pos_logprobs.values()),
                                 reverse=True)[:TOPK]
                    top10_sums.append(float(sum(math.exp(lp) for lp in lps)))

                label = int(is_correct(predicted_answer(comp.text), gold))
                labels.append(label)
                records.append({
                    "question_id": current_question_id,
                    "question": batch.iloc[qi]["question"],
                    "sample_id": si,
                    "answer_text": comp.text,
                    "label": label,                 # 1 = correct, 0 = incorrect
                    "top10_prob_sums": top10_sums,  # one value per token
                    "split": batch.iloc[qi]["split"],
                })

            # keep only questions where both outcomes occur and only store once
            if 0 < sum(labels) < N_SAMPLES:
                for r in records:
                    json.dump(r, out_f, ensure_ascii=False)
                    out_f.write("\n")
                mixed_found += 1
                mixed_bar.update(1)
                mixed_bar.set_postfix(last_qid=current_question_id,
                                      correct=f"{sum(labels)}/{N_SAMPLES}")
                used_question_ids.add(current_question_id)

    mixed_bar.close()
    scan_bar.close()
    out_f.close()
    print(f"Done. {mixed_found} mixed questions -> {OUT_PATH}")

if __name__ == "__main__":
    main()