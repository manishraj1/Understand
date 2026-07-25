"""
harness.py — shared defensive infrastructure for Phase 0.5 pilots.

House rules (StaticCore discipline):
  * Every silent failure becomes a hard assertion.
  * Prompt-template insertion is verified per-trace by substring hash.
  * Model identity, tokenizer hash, decoding params recorded in every JSON record.
  * Results checkpointed per run; a verification pass recomputes aggregates.
  * Pilot seeds are BURNED: recorded here, never reused in main runs.

Backends:
  vllm  — primary (Colab/T4).  hf — fallback.  mock — pipeline smoke test (no GPU).
"""

from __future__ import annotations
import dataclasses, hashlib, json, os, random, re, sys, time
from typing import Callable, Optional

PILOT_SEED_REGISTRY = "burned_seeds.json"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GenConfig:
    model_id: str
    backend: str                 # 'vllm' | 'hf' | 'mock'
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 4096
    seed: int = 0
    use_chat_template: bool = True   # real backends wrap prompts in the
                                     # model's chat template (R1-Distill and
                                     # Instruct models require this)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True)
                              .encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Assertions (abort loudly; never no-op)
# ----------------------------------------------------------------------------
def hard_assert(cond: bool, msg: str):
    if not cond:
        print(f"[HARD-ASSERT FAILED] {msg}", file=sys.stderr)
        raise SystemExit(2)


def assert_prompt_contains(prompt: str, fragment: str, tag: str):
    """Verify the manipulation text actually made it into the prompt."""
    hard_assert(fragment.strip() in prompt,
                f"manipulation '{tag}' missing from assembled prompt "
                f"(frag_hash={hashlib.sha256(fragment.encode()).hexdigest()[:8]})")


def burn_seeds(pilot_name: str, seeds: list[int], outdir: str):
    """Record pilot seeds so main runs can assert disjointness."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, PILOT_SEED_REGISTRY)
    reg = {}
    if os.path.exists(path):
        reg = json.load(open(path))
    reg.setdefault(pilot_name, [])
    reg[pilot_name] = sorted(set(reg[pilot_name]) | set(seeds))
    json.dump(reg, open(path, "w"), indent=2)


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
class MockBackend:
    """Deterministic canned generator for pipeline smoke tests.
    Behavior is keyed off markers in the prompt so parsers get exercised."""

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def generate(self, prompts: list[str], n: int = 1,
                 max_new_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None) -> list[list[str]]:
        out = []
        for p in prompts:
            outs = []
            for i in range(n):
                ph = int(hashlib.md5(p.encode()).hexdigest()[:8], 16)
                r = random.Random((ph ^ self.cfg.seed ^ i) & 0xFFFFFFFF)
                if "RESTATE_COST_STRUCTURE" in p:
                    # emit a mostly-correct restatement in messy R1 style
                    mt = re.search(r"reasoning inside <think> deducts ([\d\.]+) points", p)
                    think = mt.group(1) if mt else "0"
                    ma = re.search(r"final answer \(outside <think>\) deducts ([\d\.]+)", p)
                    ansc = ma.group(1) if ma else "0"
                    mw = re.search(r"wrong final answer deducts ([\d\.]+)", p)
                    wrong = mw.group(1) if mw else "0"
                    if r.random() < 0.02:
                        think = "999"
                    line = (f"CORRECT: +{think and '1000'}; THINK: -{think}"
                            f"/100tok; ANSWER: -{ansc}/word; WRONG: -{wrong}")
                    style = r.random()
                    if style < 0.4:
                        body = f"<think>rules say...</think>\nSure! **{line}**"
                    elif style < 0.7:
                        body = ("<think>let me check</think>\nThe scoring "
                                f"rules are as follows.\n{line}")
                    else:
                        body = f"CORRECT:+1000; THINK:-{think}/100tok; " \
                               f"ANSWER:-{ansc}/word; WRONG:-{wrong}"
                    outs.append(body)
                elif p.rstrip().endswith("DECISION:"):
                    # forced-decision continuation pass
                    m = re.search(r"SMOKE_ORACLE=(\w+)", p)
                    outs.append(" " + (m.group(1) if m else "TAKE"))
                elif "PANDORA" in p:
                    # near-oracle mock with R1-style format variety:
                    m = re.search(r"SMOKE_ORACLE=(\w+)", p)
                    act = m.group(1) if m and r.random() < 0.9 else \
                        r.choice(["OPEN", "TAKE"])
                    style = r.random()
                    if style < 0.25:
                        tail = f"DECISION: {act}"
                    elif style < 0.5:
                        tail = f"**DECISION: {act}**"
                    elif style < 0.75:
                        tail = f"\\boxed{{{act}}}"
                    elif style < 0.9:
                        tail = f"\\boxed{{DECISION: {act}}}"
                    else:
                        # truncation case: thinking never closes, no decision
                        outs.append("<think>DECISION: OPEN is tempting but "
                                    "let me recompute the expected val")
                        continue
                    outs.append("<think>maybe DECISION: TAKE? no wait, "
                                f"recompute {r.random():.3f}</think>\n" + tail)
                elif "AMBIG_PROBE" in p:
                    if "ambiguous" in p.lower() and r.random() < 0.15:
                        outs.append("Could you clarify which version you mean?")
                    else:
                        outs.append("Here is the answer: 42.")
                else:
                    # math-style trace with variable thinking length + answer
                    L = r.randint(40, 400)
                    ans = r.choice(["18", "18", "18", "7"])  # ~75% "pass"
                    outs.append("<think>" + ("x " * L) + "</think>\n"
                                + f"The final answer is \\boxed{{{ans}}}")
            outs and out.append(outs)
        return out


class HFBackend:
    def __init__(self, cfg: GenConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        import torch
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype="auto", device_map="auto")
        self.torch = torch

    def _wrap(self, p):
        if not self.cfg.use_chat_template:
            return p
        return self.tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, n=1, max_new_tokens=None, stop=None):
        res = []
        mnt = max_new_tokens or self.cfg.max_new_tokens
        for p in prompts:
            p = self._wrap(p)
            enc = self.tok(p, return_tensors="pt").to(self.model.device)
            outs = []
            for i in range(n):
                self.torch.manual_seed(self.cfg.seed * 100003 + i)
                o = self.model.generate(
                    **enc, do_sample=True, temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p, max_new_tokens=mnt,
                    pad_token_id=self.tok.eos_token_id)
                outs.append(self.tok.decode(o[0][enc.input_ids.shape[1]:],
                                            skip_special_tokens=True))
            res.append(outs)
        return res


class VLLMBackend:
    def __init__(self, cfg: GenConfig):
        from vllm import LLM, SamplingParams  # lazy
        self.cfg = cfg
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=cfg.model_id, dtype="auto",
                       enforce_eager=True, seed=cfg.seed)
        # HARD ASSERT: engine actually loaded the requested model
        loaded = self.llm.llm_engine.model_config.model
        hard_assert(cfg.model_id in loaded or loaded in cfg.model_id,
                    f"vLLM loaded '{loaded}' != requested '{cfg.model_id}'")

    def generate(self, prompts, n=1, max_new_tokens=None, stop=None):
        if self.cfg.use_chat_template:
            tok = self.llm.get_tokenizer()
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in prompts]
        sp = self.SamplingParams(
            n=n, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            max_tokens=max_new_tokens or self.cfg.max_new_tokens,
            stop=stop, seed=self.cfg.seed)
        outs = self.llm.generate(prompts, sp)
        return [[c.text for c in o.outputs] for o in outs]


def make_backend(cfg: GenConfig):
    return {"mock": MockBackend, "hf": HFBackend, "vllm": VLLMBackend}[cfg.backend](cfg)


# ----------------------------------------------------------------------------
# Trace utilities
# ----------------------------------------------------------------------------
BOXED = re.compile(r"\\boxed\{([^}]*)\}")
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def extract_final_answer(text: str) -> Optional[str]:
    m = BOXED.findall(text)
    if m:
        return m[-1].strip()
    m2 = re.search(r"final answer(?: is)?[:\s]*(-?[\d,\.]+)", text, re.I)
    return m2.group(1).replace(",", "").strip() if m2 else None


def thinking_tokens(text: str, approx_tokenizer: Callable[[str], int] = None) -> int:
    m = THINK.search(text)
    body = m.group(1) if m else text
    if approx_tokenizer:
        return approx_tokenizer(body)
    return max(1, len(body.split()))  # whitespace proxy; replace w/ tokenizer on rig


def normalize_numeric(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


# ----------------------------------------------------------------------------
# Checkpointing + verification
# ----------------------------------------------------------------------------
class RunLog:
    def __init__(self, outdir: str, run_name: str, cfg: GenConfig, meta: dict):
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, f"{run_name}.jsonl")
        self.records = 0
        header = {"_type": "header", "run": run_name, "ts": time.time(),
                  "config": dataclasses.asdict(cfg),
                  "config_fp": cfg.fingerprint(), "meta": meta}
        with open(self.path, "w") as f:
            f.write(json.dumps(header) + "\n")

    def write(self, rec: dict):
        rec["_type"] = "record"
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.records += 1

    def close(self, summary: dict):
        summary = dict(summary)
        summary["_type"] = "summary"
        summary["n_records"] = self.records
        with open(self.path, "a") as f:
            f.write(json.dumps(summary) + "\n")


def load_records(path: str):
    hdr, recs, summ = None, [], None
    for line in open(path):
        d = json.loads(line)
        if d["_type"] == "header":
            hdr = d
        elif d["_type"] == "record":
            recs.append(d)
        else:
            summ = d
    hard_assert(hdr is not None, f"no header in {path}")
    return hdr, recs, summ


def verify_summary(path: str, recompute: Callable[[list], dict]):
    """Recompute aggregates from raw records; assert equality with stored summary."""
    _, recs, summ = load_records(path)
    hard_assert(summ is not None, f"no summary in {path}")
    fresh = recompute(recs)
    for k, v in fresh.items():
        sv = summ.get(k)
        ok = (abs(v - sv) < 1e-9) if isinstance(v, float) else (v == sv)
        hard_assert(ok, f"summary mismatch on '{k}': stored={sv} recomputed={v}")
    print(f"[verify] {os.path.basename(path)}: all aggregates reproduce.")
