from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import Mistral3ForConditionalGeneration
except Exception:
    Mistral3ForConditionalGeneration = None

SYSTEM_PROMPT = "You are a helpful assistant that helps user compute story similarity"

PROMPT = """Task Description
In this study, you are tasked with identifying similar stories.

In each annotation, you will be presented with three stories, an anchor, and two choices, a and b. You are to determine which of the candidate stories, a and b, is the most similar to the anchor and provide a similarity score between 0 and 100. The similarity between a and b is irrelevant.

In each step, you select the candidate story that is more similar to the anchor. Specifically, you will consider the stories' narrative similarity. Additionally, you are asked to specify which general aspects of similarity the two stories share.

Narrative Similarity
The narrative similarity of stories can be broken down into three core aspects: (1) the abstract themes of the story, (2) the course of action, and (3) the story outcomes. At one extreme, this means that the story deals with the same themes and tells the same order of events with an identical outcome or conclusion, just using a different wording; at the other extreme, the story might be completely different and lack any basis for comparison.

More difficult to assess are stories that only share some similarities. In such cases, you are asked to weigh the three core components of story similarity. You should focus on the core aspects of stories, potentially largely ignoring side storylines. How you weigh the individual factors should be based on your intuitive impression of which aspects you consider crucial to the overall similarity of the specific stories.

We define these three aspects as follows:
- Abstract Theme describes the defining constellation of problems, central ideas, and core motifs of a story. The definition does not cover the concrete setting of a story.
- Course of Action describes sequences of events, actions, conflicts, and turning points in a story and the order in which they happen.
- Outcomes describe the results of the plot at the end of the text, for example, the conflict resolution, the characters' fates, moral lessons, etc. It does not cover intermediate statuses that change later in the story.

Each aspect can take different forms in an actual pair of stories. Below, we list one example for each aspect:
- The general setting of the story, if it strongly influences the events in the story or the events necessitate a specific setting (abstract themes)
    - A: On the week-long journey from Europe to the Americas, the crew members get into a heated conflict about the best ration packages.
    - B: The flight to Mars is long. After several weeks, the astronauts become better friends than ever before, having to share the limited resources.
    - A and B share some similarities in that the polar opposite outcomes are both enabled by being cut off from the outside world.

- The order of events in the story (course of action)
    - A: After the ship capsizes and Alice barely makes it out alive, she starts living life to the fullest.
    - B: Alice is living life to the fullest until, one day, her ship capsizes. She barely makes it out alive.
    - A and B are similar in that both tell of a good life and a shipwreck (abstract theme), but they differ in the course of action, and the order is very different.

- The outcomes of events (story outcomes)
    - A: The man intentionally drops a cup; it breaks.
    - B: He accidentally swipes the bottle off the table, and it shatters.
    - A and B are similar in that the events are comparable and lead to similar outcomes.

There is a range of factors that expressly do NOT contribute to the narrative similarity:
    - The style of writing in a story
    - The concrete setting of a story (also including the time period).
    - The names of the characters and locations
    - The length of a text
    - The level of detail in which the events are told.

Differentiating Between Similarity Aspects
Distinguishing the three aspects can be challenging. In general, it is important to consider each aspect independently.

Often, pairs of stories that are similar in terms of course of action will also share an abstract theme. However, it is possible that similar events emerge from completely different surrounding circumstances. Outcomes, on the other hand, are clearly distinct from the other two aspects: practically identical events in stories with comparable abstract themes can result in polar opposite outcomes.

When comparing abstract themes, it can help to explicitly formulate them. There is, of course, no single correct answer, and a single story's theme could be formulated in many ways. Two stories share a general theme if there is a description that captures the defining circumstances of both stories.


Example 1:
Anchor: Anna loses her purse. She is terrified because there are important documents in it. She retraces her steps but cannot find it. Dan finds it and helpfully returns it to her.
Text A: Brian lost his backpack. He did not care too much, as only a water bottle was in it. After an hour of searching, he finally found it.
Text B: Alex loses his engagement ring while swimming. He freaks out, and after hours of diving for it, he still cannot find it.
Solution: {"chosen_text": "A", "similarity_A": 60, "similarity_B": 40, "explanation": "A and Anchor tell the story of a lost item that is retrieved. In the case of A, it is found by a third party (as it is in the Anchor), while in B it is not found at all."}


Example 2:
Anchor: Anna loses her purse. She retraces her steps but cannot find it. Dan finds it and helpfully returns it to her.
Text A: Brian lost his backpack. He was terrified because there were important documents in it. After an hour of intense search he finally found it.
Text B: Alex lost his engagement ring while swimming. After hours of looking, he still can not find it. Karen finds the ring while magnet fishing and, based on the engraved name, manages to return it.
Solution: {"chosen_text": "B", "similarity_A": 60, "similarity_B": 85, "explanation": "Now, B adds an ending where a third party finds the lost item, which means that the outcomes align in both cases. In this case, the third party finding the item is the decisive factor in identifying the more similar story."}


Example 3:
Anchor: In the trenches of World War I, Greg is hit by a grenade splinter. He is in tremendous pain, but his comrades manage to evacuate him from the position. After spending weeks fighting the infection in his leg, he succumbs to his injuries.
Text A: Jill was driving home when another car suddenly crashed into hers. After receiving medical attention, she recovered within just days and now advocates for traffic safety.
Text B: As Major Miller gives the command to charge, he is not sure if his men can manage it. In a heroic effort, they capture the next position. Only one day later, though, it is again lost to the enemy.
Solution: {"chosen_text": "A", "similarity_A": 35, "similarity_B": 10, "explanation": "While the concrete setting in Anchor and B is a battlefield, the similarities end there. A by contrast and B both deal with recovery from injuries as an abstract theme. They do not share a similar outcome, and the similarity in the course of action is also limited."}


Example 4:
Anchor: Maven is a magician; ever since finishing his apprenticeship, he has worked on developing novel magic in his tower. As the years go on, he gets fewer and fewer visitors, focusing his life only on his work. After he dies alone in his tower, nobody finds his body for many decades.
Text A: The expedition is not going as planned. Some party members have abandoned the mission and tried to return home. One day, the other three remaining members decide to head home, and Ellie realizes she is the last one in the expedition party. Working through snow and ice, she underestimates the storm and freezes to death; nobody ever finds her.
Text B: Three friends go on a fishing trip. They catch nothing. They still had a great time.
Solution: {"chosen_text": "A", "similarity_A": 25, "similarity_B": 0, "explanation": "Neither option presents a similar course of action or theme. There is however a similar outcome in Anchor and A."}


Example 5:
Anchor: Andrew goes to the shop to buy food and drinks. He then heads home and prepares everything for his family's arrival. As aunts and uncles arrive, he can impress them with homemade cookies and fancy drinks.
Text A: Zoie buys ammunition and guns; she will need them. Back home, she prepares well, setting up traps and protected firing positions. When the Zombies rush her doors, she is prepared and can deal out destruction. Nonetheless, she cannot win against the unending hoards of undead.
Text B: Erica is great at building paper planes. One day, to her surprise, she attends a competition, and despite little preparation, she wins!
Solution: {"chosen_text": "A", "similarity_A": 20, "similarity_B": 5, "explanation": "While the abstract theme and outcome of Anchor and A have little to do with each other, they both describe a process of first purchasing something, then preparing something, and finally making use of the preparations. This is a case of a similar course of action."}


Strictly adhere to the desired JSON format with nothing else and solve for:
"""

DATA_PROMPT = """Anchor: {0}
Text A: {1}
Text B: {2}"""


def _safe_model_suffix(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name.split("/")[-1]).strip("-")


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_output_path(dataset_path: Path, model_name: str, output_path: str) -> Path:
    if output_path:
        return Path(output_path)
    suffix = _safe_model_suffix(model_name)
    return dataset_path.with_name(f"{dataset_path.stem}_{suffix}.jsonl")


def build_message_and_prompt(
    model_name: str,
    tokenizer: Any,
    entry: dict,
    device: torch.device,
) -> tuple[Any, Any]:
    full_prompt = PROMPT + DATA_PROMPT.format(
        _normalize_text(entry.get("anchor_text")),
        _normalize_text(entry.get("text_a")),
        _normalize_text(entry.get("text_b")),
    )
    attention_mask = None
    model_l = model_name.lower()

    if "qwen3" in model_l or "gemma-2" in model_l or "gpt-oss" in model_l:
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + full_prompt}]
    elif "gemma-3" in model_l:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": full_prompt}]},
        ]
    elif "mistral-small-3." in model_l:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": full_prompt}]},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ]

    if "gemma-3" in model_l:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            tokenize=True,
        )
    elif "mistral-small-3." in model_l:
        try:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except Exception as exc:
            raise RuntimeError(
                "mistral-common is required for mistral-small-3.* prompting."
            ) from exc
        if not isinstance(tokenizer, MistralTokenizer):
            raise RuntimeError("Tokenizer type mismatch for mistral-small-3.*")
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=messages)
        )
        input_ids = torch.tensor([tokenized.tokens], device=device)
        attention_mask = torch.ones_like(input_ids)
    elif "gpt-oss" in model_l:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_special_tokens=True,
            add_generation_prompt=True,
        ).to(device)

    return input_ids, attention_mask


def _postprocess_model_output(model_name: str, text: str) -> str:
    if "Mistral-Small-24B" in model_name and "[/INST]" in text:
        return text.split("[/INST]", 1)[1].strip()
    if "oss" in model_name.lower():
        start = "<|end|><|start|>assistant<|channel|>final<|message|>"
        end = "<|return|>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0].strip()
    return text.strip()


def run_on_gpu(
    data: list[dict],
    model_name: str,
    output_path: Path,
    *,
    log_every: int,
    max_new_tokens_default: int,
    max_new_tokens_mistral: int,
    torch_dtype: str,
    device_map: str,
) -> None:
    model_l = model_name.lower()

    if "mistral-small-3." in model_l:
        if Mistral3ForConditionalGeneration is None:
            raise RuntimeError(
                "Mistral3ForConditionalGeneration is unavailable in this transformers version."
            )
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except Exception as exc:
            raise RuntimeError(
                "mistral-common is required for mistral-small-3.* models."
            ) from exc
        tokenizer = MistralTokenizer.from_hf_hub(model_name)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()
    else:
        if "gemma-3" in model_l:
            tokenizer = AutoProcessor.from_pretrained(model_name)
            decode_tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            decode_tokenizer = tokenizer

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype_map[torch_dtype],
        ).eval()

    if "mistral-small-3." in model_l:
        decode_tokenizer = None

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    local_data: list[dict] = []
    t0 = time.time()
    every = max(log_every, 1)
    for entry_id, entry in enumerate(data):
        if entry_id % every == 0:
            print(
                "Processing entry {0}/{1}, time so far: {2:.2f}s".format(
                    entry_id, len(data), time.time() - t0
                ),
                flush=True,
            )

        input_ids, attention_mask = build_message_and_prompt(
            model_name, tokenizer, entry, model_device
        )

        with torch.inference_mode():
            if "mistral-small-3." in model_l:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens_mistral,
                )
            else:
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens_default,
                )

        if "mistral-small-3." in model_l:
            if not hasattr(tokenizer, "decode"):
                raise RuntimeError("Mistral tokenizer does not implement decode().")
            out = tokenizer.decode(outputs[0])
        else:
            if decode_tokenizer is None:
                raise RuntimeError("Missing decode tokenizer.")
            out = decode_tokenizer.decode(outputs[0])
        out = _postprocess_model_output(model_name, out)

        local_data.append(
            {
                "anchor_text": _normalize_text(entry.get("anchor_text")),
                "text_a": _normalize_text(entry.get("text_a")),
                "text_b": _normalize_text(entry.get("text_b")),
                "model_output": out,
            }
        )

    print(
        "Processing entry {0}/{1}, time so far: {2:.2f}s".format(
            len(data), len(data), time.time() - t0
        ),
        flush=True,
    )
    _write_jsonl(output_path, local_data)
    print("Done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Track A LLM outputs (anchor/text_a/text_b -> model_output)."
    )
    parser.add_argument("--model-name", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--dataset-path",
        default="miscellaneous/datasets/Narrative Similarity Task/development/dev_track_a.jsonl",
    )
    parser.add_argument("--output-path", default="")
    parser.add_argument("--hf-home", default="/export/projects/nlp/.cache/")
    parser.add_argument("--max-new-tokens", type=int, default=131072)
    parser.add_argument("--max-new-tokens-mistral", type=int, default=65536)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    dataset_path = Path(args.dataset_path)
    output_path = _resolve_output_path(dataset_path, args.model_name, args.output_path)
    rows = _load_jsonl(dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"GPUS {torch.cuda.device_count()}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {output_path}")

    run_on_gpu(
        rows,
        args.model_name,
        output_path,
        log_every=args.log_every,
        max_new_tokens_default=args.max_new_tokens,
        max_new_tokens_mistral=args.max_new_tokens_mistral,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
