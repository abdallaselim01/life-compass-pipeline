"""
Stage fingerprints: which prompt text produced which stage of a video, and what to redo when a prompt changed.

Each stage has a fingerprint = SHA-256[:16] of its prompt file(s), read as text with CRLF turned into LF (so a
change of line endings alone never redoes a stage; for an LF file the value equals the `sha256[:16]` the run log
prints for it). A stage's EFFECTIVE fingerprint also covers everything before it:

    effective(listen)  = prompt_fp(listen)
    effective(stage)   = sha16(prompt_fp(stage) + ":" + effective(previous stage))

so a new listen prompt changes the effective fingerprint of extract too, while a new extract prompt leaves listen's
alone. `plan()` compares what is recorded for a video with what the prompts are now and returns the first stage to
redo (that stage and every later one) and the stages whose stored result is reused.

A stage's own contribution is its prompt AND its API settings (llm.request_settings): sha16(prompt_fp + ":" + settings_fp).
A change of either one redoes the stage and every stage after it. The model name is NOT part of it (it is chosen at run time by
the fallback chain). Two more stages run after the principles exist -- match, caption (see post_current / post_plan):
each has its own prompt and settings, caption covers match, and both cover the last principle stage.

No database, no model call, no network. The settings are read from llm (imported inside settings_fp, so the module itself
stays importable without the model library).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# stage -> prompt files under prompts/ (a stage that reads several is fingerprinted over all of them): the system prompt and the
# files that make up the request (everything the model reads besides the audio / the transcript itself)
STAGE_PROMPTS = {
    "listen": ("transcribe_prompt.txt", "listen_request.txt", "listen_retry_note.txt"),
    "extract": ("extract_prompt.txt", "extract_request.txt"),
}
# the principle stages, in the order they run
PRINCIPLE_STAGES = ("listen", "extract")
# the stages that run after the principles exist (matching of the video's images to its principles, captions of the images
# left unmatched); recorded per video in the same table as the stages above
POST_STAGES = ("match", "caption")


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def settings_fp(stage: str) -> str:
    """Fingerprint of the API settings the stage sends (llm.request_settings)."""
    from . import llm
    return llm.settings_fp(stage)


def with_settings(stage: str, prompt_fingerprint: str) -> str:
    """A stage's own contribution to its effective fingerprint: its prompt and its API settings."""
    return sha16(f"{prompt_fingerprint}:{settings_fp(stage)}")


def normalized_text(path: Path) -> str:
    """The file as text, CRLF -> LF (nothing else is touched)."""
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def prompt_fp(stage: str, root: Path | None = None) -> str:
    """Fingerprint of the prompt text of one stage."""
    folder = Path(root or PROJECT_ROOT) / "prompts"
    parts = [sha16(normalized_text(folder / name)) for name in STAGE_PROMPTS[stage]]
    return parts[0] if len(parts) == 1 else sha16("\n".join(parts))


def chain(prompt_fingerprint: str, previous_effective: str | None) -> str:
    """Effective fingerprint of a stage, given the effective fingerprint of the stage before it (None for the first)."""
    if previous_effective is None:
        return prompt_fingerprint
    return sha16(f"{prompt_fingerprint}:{previous_effective}")


@dataclass(frozen=True)
class StageFp:
    prompt_fp: str
    effective_fp: str
    settings_fp: str = ""


def current_fingerprints(root: Path | None = None) -> dict:
    """{stage: StageFp} for the principle stages, from the prompt files as they are right now."""
    out, prev = {}, None
    for stage in PRINCIPLE_STAGES:
        pfp = prompt_fp(stage, root)
        sfp = settings_fp(stage)
        eff = chain(sha16(f"{pfp}:{sfp}"), prev)
        out[stage] = StageFp(pfp, eff, sfp)
        prev = eff
    return out


def effective_chain(prompt_fps: dict, start_stage: str, stored: dict) -> dict:
    """{stage: (prompt_fp, effective_fp)} for the stages a run starting at `start_stage` will produce.
    `prompt_fps` is {stage: prompt_fp} as read at the start of the run; `stored` is {stage: effective_fp} of what
    is recorded for the video. A run that starts after the first stage builds on the recorded effective
    fingerprint of the stage before it; when that is not recorded, nothing can be said and {} is returned."""
    order = PRINCIPLE_STAGES
    if start_stage not in order:
        return {}
    i0 = order.index(start_stage)
    prev = None
    if i0 > 0:
        prev = stored.get(order[i0 - 1])
        if prev is None:
            return {}
    out = {}
    for stage in order[i0:]:
        eff = chain(with_settings(stage, prompt_fps[stage]), prev)
        out[stage] = (prompt_fps[stage], eff)
        prev = eff
    return out


@dataclass
class Plan:
    rerun_from: str | None = None            # None: nothing to redo
    rerun: tuple = ()
    reuse: tuple = ()
    reasons: dict = field(default_factory=dict)      # stage -> why it is stale ("changed" | "missing" | "follows <stage>")
    kinds: dict = field(default_factory=dict)        # stage -> what changed when reasons says "changed": "prompt" | "settings"
    stored: dict = field(default_factory=dict)
    current: dict = field(default_factory=dict)

    @property
    def needed(self) -> bool:
        return self.rerun_from is not None

    def reason_text(self) -> str:
        if not self.needed:
            return "nothing to redo"
        first = self.rerun_from
        why = self.reasons[first]
        if why == "changed":
            what = "API settings" if self.kinds.get(first) == "settings" else "prompt"
            return (f"{what} of '{first}' changed (recorded {self.stored.get(first)} -> now "
                    f"{self.current[first].effective_fp}); redo {' -> '.join(self.rerun)}")
        return f"no fingerprint recorded for '{first}'; redo {' -> '.join(self.rerun)}"

    def note(self) -> str:
        """The line added to the Telegram message of a redone video."""
        if not self.needed:
            return ""
        if self.reasons[self.rerun_from] == "changed":
            if self.kinds.get(self.rerun_from) == "settings":
                return f"Reprocessing: the API settings of stage {self.rerun_from} changed"
            return f"Reprocessing: the prompt of stage {self.rerun_from} changed"
        return f"Reprocessing: stage {self.rerun_from} has no recorded fingerprint"

    def lines(self) -> list:
        out = ["STAGE PLAN: "
               + ("nothing to redo" if not self.needed else f"redo from '{self.rerun_from}'")]
        for stage in self.current:
            rec = self.stored.get(stage)
            cur = self.current[stage].effective_fp
            verdict = ("reuse stored result" if stage in self.reuse else f"REDO ({self.reasons.get(stage)})")
            out.append(f"  {stage}: recorded={rec or '-'} now={cur} -> {verdict}")
        return out


def plan(stored: dict, current: dict, details: dict | None = None) -> Plan:
    """`stored` = {stage: effective_fp} recorded for the video; `current` = current_fingerprints().
    The first stage whose recorded effective fingerprint is missing or different is redone, and every later one.
    `details` = {stage: {"prompt_fp", "settings_fp", ...}} as recorded (store.get_stage_fingerprints): only used to say
    whether the prompt or the API settings of the first changed stage changed."""
    order = PRINCIPLE_STAGES
    result = Plan(stored=dict(stored), current=dict(current))
    first = None
    for stage in order:
        rec = stored.get(stage)
        if rec is None:
            first, why = stage, "missing"
        elif rec != current[stage].effective_fp:
            first, why = stage, "changed"
            result.kinds[stage] = _change_kind((details or {}).get(stage), current[stage])
        else:
            continue
        result.reasons[stage] = why
        break
    if first is None:
        result.reuse = tuple(order)
        return result
    i = order.index(first)
    result.rerun_from = first
    result.rerun = tuple(order[i:])
    result.reuse = tuple(order[:i])
    for later in order[i + 1:]:
        result.reasons[later] = f"follows {first}"
    return result


def _change_kind(recorded: dict | None, now: StageFp) -> str:
    """"prompt" when the recorded prompt differs (or nothing is known); otherwise the settings changed (a row from before
    settings were recorded has an empty settings_fp and counts as changed once)."""
    if not recorded or recorded.get("prompt_fp") != now.prompt_fp:
        return "prompt"
    return "settings"


# ---- the stages after the principles: match, caption ---------------------------------------------------------
def post_current(upstream_effective: str, prompt_fps: dict) -> dict:
    """{stage: StageFp} of match / caption for one video, as they would be recorded now.
    `upstream_effective` = effective fingerprint of the last principle stage of the video; `prompt_fps` = {stage: prompt
    fingerprint}.
      match   = own(match) after the principles
      caption = own(caption) after match
    where own(stage) = prompt + API settings of the stage. (Until 2026-11-10 match also covered the describe fingerprints of
    the images; images are no longer described, and a picture that is dropped again is replaced and matched again directly.)"""
    own = {s: with_settings(s, prompt_fps[s]) for s in POST_STAGES}
    match = chain(own["match"], upstream_effective)
    caption = chain(own["caption"], match)
    return {s: StageFp(prompt_fps[s], eff, settings_fp(s))
            for s, eff in (("match", match), ("caption", caption))}


@dataclass
class PostPlan:
    redo: tuple = ()                                  # the post stages to redo, in order
    kinds: dict = field(default_factory=dict)         # stage -> "missing" | "prompt" | "settings" | "upstream"

    def reason_text(self) -> str:
        names = {"missing": "no fingerprint recorded", "prompt": "prompt changed", "settings": "API settings changed",
                 "upstream": "an earlier stage changed"}
        return "; ".join(f"{s}: {names[self.kinds[s]]}" for s in self.redo)


def post_plan(stored: dict, current: dict, details: dict | None = None) -> PostPlan:
    """`stored` = {stage: effective_fp} recorded for the video's post stages; `current` = post_current(...). A stage is redone
    when its recorded effective fingerprint is missing or different. Because match is part of caption's fingerprint, a redone
    match redoes the caption as well."""
    redo, kinds = [], {}
    for stage in POST_STAGES:
        now = current[stage]
        rec = stored.get(stage)
        if rec is None:
            kind = "missing"
        elif rec == now.effective_fp:
            continue
        else:
            d = (details or {}).get(stage) or {}
            if d and d.get("prompt_fp") != now.prompt_fp:
                kind = "prompt"
            elif d and d.get("settings_fp", "") != now.settings_fp:
                kind = "settings"
            else:
                kind = "upstream"
        redo.append(stage)
        kinds[stage] = kind
    return PostPlan(tuple(redo), kinds)
