# Fine-Tuning GPT-2 for Python Code Generation

Fine-tuned GPT-2 (124M and 355M) to turn natural-language instructions into Python
functions, and built an execution-based evaluation harness that runs the generated
code against real test cases instead of relying on exact-string matching.

```
Instruction:  "Write a Python function that reverses a linked list"

Generated:     def reverse_list(head):
                   prev = None
                   while head:
                       nxt = head.next
                       head.next = prev
                       prev = head
                       head = nxt
                   return prev
```

---

## The problem

GPT-2 is a next-token predictor, not an instruction-following model like ChatGPT.
Ask the base model to "write a function" and it just continues the sentence as plain
text. The goal of this project was to fine-tune it so an instruction reliably produces
the matching Python code, and then to measure how well that actually works.

The approach was to format training data as `instruction → code` pairs so the model
learns the pattern, fine-tune GPT-2 small and medium on it, and evaluate the output by
actually executing it.

## Approach

- **Base model:** GPT-2 small (124M) and medium (355M), loaded with pretrained weights
- **Training:** instruction fine-tuning (AdamW, cross-entropy on next-token prediction),
  run on an NVIDIA RTX 2060 (CUDA)
- **Dataset:** a curated set of ~1,000 Python functions balanced across categories
  (arithmetic, strings, lists, sorting, searching, linked lists, trees, graphs, dynamic
  programming), later merged with ~2,500 real GitHub functions for stylistic variety
- **Generation:** prompt with the instruction template, stop at the `<|endoftext|>`
  token (`eos_id = 50256`)
- **Evaluation:** a custom harness that runs each generated function against test cases
  and records whether it passes, fails, crashes, or is untestable

## Results

Evaluating on held-out algorithm tasks, **execution-based accuracy told a very
different story than exact-match** — the model writes plenty of correct code that simply
isn't byte-identical to the reference.

| Model | Exact Match | Functional Accuracy (executed) |
|-------|:-----------:|:------------------------------:|
| GPT-2 Small (124M) | 21% | 23% |
| GPT-2 Medium (355M) | 37% | **65%** |

Breaking the medium model's executed output into outcomes: **65% pass** (correct output),
only **2% fail** (runs but wrong), and **33% error** (crashes or incomplete). When it is
right, it is genuinely right — its failures are mostly incomplete code rather than subtle
logic bugs.

## Key findings

**Data balance mattered more than model size.** On an arithmetic-heavy dataset, scaling
from small to medium barely helped. After rebalancing the dataset so complex categories
(linked lists, trees, graphs) were well represented, the same jump produced a real gain.
Extra model capacity only pays off when the data supports it.

**Structure is learned before logic.** Fine-tuning quickly teaches the *shape* of code —
class definitions, function signatures, indentation — before it learns the underlying
algorithm. The model would produce a perfectly-formed `ListNode` class and function
signature, then fill the body with logic that didn't actually work.

**The evaluation metric shapes the conclusion.** Exact-match made the model look mediocre
and scored real-world functions at 0% even when they were reasonable, because they
reference context from their original files. Switching to execution-based testing revealed
the medium model was nearly 3× more capable than exact-match suggested. Measuring
correctness by running the code, the way real code-generation research does, changed the
whole picture.

## Repository structure

```
.
├── data/
│   ├── code-functions-balanced.json   # curated, category-balanced dataset
│   └── code-functions-merged.json     # curated + real GitHub functions
├── src/
│   ├── finetune.py                    # instruction fine-tuning script
│   ├── generate.py                    # inference / text generation
│   └── eval_harness.py                # execution-based evaluation
└── results/
    ├── responses/                     # model outputs (JSON)
    └── loss_plots/                    # training curves
```

## Running it

Fine-tune a model (set `CHOOSE_MODEL` and the dataset path inside the script):

```bash
python src/finetune.py
```

Evaluate a set of generated responses by executing them against test cases:

```bash
python src/eval_harness.py results/responses/medium_balanced.json
```

The harness reports, for every testable function, whether it **passes** (correct output),
**fails** (runs but wrong), **errors** (crashes/times out), or is **untestable**.

## What I'd do next

- Expand the execution harness with more test cases per category for tighter scores
- Sweep generation temperature and top-k to study the reliability-vs-creativity tradeoff
- Scale to GPT-2 large now that the data clearly supports more capacity
- Curate self-contained real functions so they can be both trained on and auto-tested

## Built on

Sebastian Raschka's *Build a Large Language Model From Scratch* (chapters 5 and 7) for the
GPT-2 implementation and instruction-tuning loop, with a custom dataset, generation
helpers, and evaluation harness.
