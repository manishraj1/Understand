"""
prompts.py — every manipulation text in one reviewable file.

These strings ARE the experiment. Change nothing after freeze without a
DEVIATIONS.md entry. Each condition has a tag used for per-trace insertion
assertions (harness.assert_prompt_contains).

Framing decision (Document B §5.3 + open question OQ1): costs are framed as
POINTS deducted from a task score the model is told it is maximizing —
incentive-compatible-ish, avoids pretending the model has money. Dollar and
token-budget framings are OQ1 variants, defined here for the supplementary arm.
"""

# ---------------------------------------------------------------------------
# Base task wrapper (compute arm, Exp 1). {question} filled per item.
# ---------------------------------------------------------------------------
TASK_HEADER = (
    "You are solving a math problem. Reason step by step inside <think> tags, "
    "then give your final answer as \\boxed{{answer}}.\n"
)

SCORE_PREAMBLE = (
    "SCORING: You are maximizing a task score. A correct final answer earns "
    "+1000 points.\n"
)

# --- Computation-cost doses (log-spaced), score-deduction framing -----------
CC_DOSES = {
    "cc_low":  "THINKING COST: every 100 tokens of reasoning inside <think> "
               "deducts 0.01 points from your score.\n",
    "cc_mid":  "THINKING COST: every 100 tokens of reasoning inside <think> "
               "deducts 1 points from your score.\n",
    "cc_high": "THINKING COST: every 100 tokens of reasoning inside <think> "
               "deducts 100 points from your score.\n",
}

# --- Error-cost doses (fixed modest compute cost, graded stakes) -------------
CE_FIXED_CC = ("THINKING COST: every 100 tokens of reasoning inside <think> "
               "deducts 1 points from your score.\n")
CE_DOSES = {
    "ce_low":  "ERROR COST: a wrong final answer deducts 1 points from your "
               "score.\n",
    "ce_mid":  "ERROR COST: a wrong final answer deducts 100 points from your "
               "score.\n",
    "ce_high": "ERROR COST: a wrong final answer deducts 10000 points from "
               "your score.\n",
}

# --- Controls ----------------------------------------------------------------
EMPTY_COST = ("NOTE: tokens are billed at the standard rate. This does not "
              "affect your score.\n")

BREVITY = "Think concisely.\n"

PLACEBO_DOSES = {   # graded cost attached to near-constant quantity
    "pl_low":  "ANSWER COST: every word in your final answer (outside <think>) "
               "deducts 0.01 points from your score.\n",
    "pl_mid":  "ANSWER COST: every word in your final answer (outside <think>) "
               "deducts 1 points from your score.\n",
    "pl_high": "ANSWER COST: every word in your final answer (outside <think>) "
               "deducts 100 points from your score.\n",
}

# --- OQ1 framing variants (single dose = cc_mid magnitude) -------------------
OQ1_FRAMINGS = {
    "dollar":  "THINKING COST: every 100 tokens of reasoning costs $0.10, "
               "charged against your task budget.\n",
    "budget":  "THINKING BUDGET: you have a soft budget of 800 thinking "
               "tokens; exceeding it reduces your score proportionally.\n",
    "points":  CC_DOSES["cc_mid"],
}

# ---------------------------------------------------------------------------
# K0b comprehension probe — RESTATE_COST_STRUCTURE marker is load-bearing
# ---------------------------------------------------------------------------
K0B_PROBE = (
    "RESTATE_COST_STRUCTURE: Do NOT solve the problem below. Your only task "
    "is to restate the scoring rules of this task. You may reason briefly, "
    "but you must END your response with exactly one line of this form, "
    "using digits only (write 0 for any component the rules do not "
    "specify):\n"
    "CORRECT:+<n>; THINK:-<n>/100tok; ANSWER:-<n>/word; WRONG:-<n>\n"
)

# ---------------------------------------------------------------------------
# P0 Pandora's Box (one-step capability control). PANDORA marker load-bearing.
# {fallback} {v} {p} {c} filled per instance. SMOKE_ORACLE only in mock mode.
# ---------------------------------------------------------------------------
PANDORA_TEMPLATE = (
    "PANDORA decision task. You currently hold a guaranteed prize of "
    "{fallback} points.\nThere is one sealed box. With probability {p} it "
    "contains {v} points; otherwise it contains 0 points.\nOpening the box to "
    "look inside costs {c} points (paid regardless of contents). After "
    "opening, you may keep the better of the box contents or your guaranteed "
    "prize.\nDecide: reply with exactly 'DECISION: OPEN' to pay and open, or "
    "'DECISION: TAKE' to keep the guaranteed prize without opening.\n"
    "Maximize expected points. Think step by step first.\n"
)

# ---------------------------------------------------------------------------
# P2 clarification probes. AMBIG_PROBE marker load-bearing.
# ---------------------------------------------------------------------------
P2_SYSTEM_BASE = ("You are a helpful assistant.")
P2_SYSTEM_PERMISSION = (
    "You are a helpful assistant. If a request is ambiguous, you may ask one "
    "clarifying question before answering.")

P2_ITEMS = [
    # (id, variant, text)  — 8 seed items; expand to 60 in Exp 2 proper.
    ("i01", "unambiguous", "AMBIG_PROBE: Convert 25 kilometers to miles."),
    ("i01", "ambiguous",   "AMBIG_PROBE: Convert 25 to miles."),
    ("i02", "unambiguous", "AMBIG_PROBE: Write a Python function that sorts a "
                            "list of integers ascending."),
    ("i02", "ambiguous",   "AMBIG_PROBE: Write a Python function that sorts "
                            "my data."),
    ("i03", "unambiguous", "AMBIG_PROBE: Summarize the plot of Hamlet in two "
                            "sentences."),
    ("i03", "ambiguous",   "AMBIG_PROBE: Summarize it in two sentences."),
    ("i04", "unambiguous", "AMBIG_PROBE: Book me a table for 2 at an Italian "
                            "restaurant tonight at 7pm — draft the request "
                            "message."),
    ("i04", "ambiguous",   "AMBIG_PROBE: Book me a table tonight — draft the "
                            "request message."),
]


def build_exp1_prompt(condition: str, question: str) -> tuple[str, str]:
    """Returns (prompt, load_bearing_fragment) for insertion assertion."""
    if condition == "baseline":
        frag = TASK_HEADER
        body = TASK_HEADER
    elif condition == "empty":
        frag = EMPTY_COST
        body = TASK_HEADER + SCORE_PREAMBLE + EMPTY_COST
    elif condition == "brevity":
        frag = BREVITY
        body = TASK_HEADER + BREVITY
    elif condition in CC_DOSES:
        frag = CC_DOSES[condition]
        body = TASK_HEADER + SCORE_PREAMBLE + frag
    elif condition in CE_DOSES:
        frag = CE_DOSES[condition]
        body = TASK_HEADER + SCORE_PREAMBLE + CE_FIXED_CC + frag
    elif condition in PLACEBO_DOSES:
        frag = PLACEBO_DOSES[condition]
        body = TASK_HEADER + SCORE_PREAMBLE + frag
    elif condition in OQ1_FRAMINGS:
        frag = OQ1_FRAMINGS[condition]
        body = TASK_HEADER + SCORE_PREAMBLE + frag
    else:
        raise ValueError(f"unknown condition {condition}")
    return body + "\nPROBLEM: " + question + "\n", frag


ALL_EXP1_CONDITIONS = (
    ["baseline", "empty", "brevity"]
    + list(CC_DOSES) + list(CE_DOSES) + list(PLACEBO_DOSES)
)
