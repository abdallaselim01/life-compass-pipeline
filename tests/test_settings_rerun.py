"""A change of the API settings of a stage redoes that stage and every stage after it, exactly like a change of its prompt
(design decision 2026-11-06). Covers the one source of the settings, the fingerprints of the principle stages and of the stages after
them (match / caption), the plan, the migration to schema v5 and the bookmarklet path. No model is ever called."""
import sqlite3

import pytest
from google.genai import types

from lc_test_helpers import get_image
from lifecompass import enrichment, llm, pipeline, store
from lifecompass import stage_fingerprints as sf

# fixtures and helpers of the sibling test modules
from test_image_replace import URL, VID, folders, pipe, rows, seed_video  # noqa: F401
from test_stage_rerun import env, extract_ok, heard, seed  # noqa: F401

JSON_STAGES = ("extract", "match", "caption")


def quiet(*args, **kwargs):
    return None


# ---- one source of the settings ---------------------------------------------------------------------------------
def test_the_config_that_is_sent_is_built_from_the_settings_dict_of_each_stage():
    for stage in llm.STAGES:
        s = llm.request_settings(stage)
        cfg = types.GenerateContentConfig(**llm.max_quality_config_kwargs(stage))
        assert cfg.thinking_config.thinking_level == getattr(types.ThinkingLevel, s["thinking_level"])
        assert cfg.max_output_tokens == s["max_output_tokens"]
        assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None
    base = types.GenerateContentConfig(**llm.max_quality_config_kwargs())          # what generate_text / json / audio send
    for stage in llm.STAGES:
        cfg = types.GenerateContentConfig(**llm.max_quality_config_kwargs(stage))
        assert (cfg.thinking_config.thinking_level, cfg.max_output_tokens) == (base.thinking_config.thinking_level,
                                                                              base.max_output_tokens)


def test_only_the_stages_that_send_a_picture_have_a_media_resolution_and_the_model_is_never_a_setting():
    assert "describe" not in llm.STAGES
    for stage in ("match", "caption"):
        assert llm.request_settings(stage)["media_resolution"] == "ULTRA_HIGH"
        assert llm.media_resolution_level(stage) == types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
    assert all("media_resolution" not in llm.request_settings(s) for s in llm.STAGES if s not in ("match", "caption"))
    assert all("model" not in llm.request_settings(s) for s in llm.STAGES)


def test_the_settings_fingerprint_is_stable_and_follows_every_setting(monkeypatch):
    base = {s: llm.settings_fp(s) for s in llm.STAGES}
    assert base == {s: llm.settings_fp(s) for s in llm.STAGES}
    monkeypatch.setattr(llm, "_MAX_OUTPUT_TOKENS", 1000)
    assert all(llm.settings_fp(s) != base[s] for s in llm.STAGES)
    monkeypatch.setattr(llm, "_MAX_OUTPUT_TOKENS", 65536)
    monkeypatch.setattr(llm, "_SCHEMA_ENABLED", True)           # only the stages that answer in JSON are affected
    for s in llm.STAGES:
        assert (llm.settings_fp(s) != base[s]) == (s in JSON_STAGES)


def test_the_enrichment_calls_send_the_settings_of_their_own_stage():
    import inspect
    src = inspect.getsource(enrichment)
    for stage in ("match", "caption"):
        assert f'max_quality_config_kwargs("{stage}")' in src
    assert "max_quality_config_kwargs()" not in src
    assert 'media_resolution_level("match")' in src and 'media_resolution_level("caption")' in src


# ---- principle stages -------------------------------------------------------------------------------------------
def test_a_change_of_settings_changes_that_stage_and_the_ones_after_it_but_not_the_ones_before(monkeypatch):
    base = sf.current_fingerprints()
    real = sf.settings_fp
    monkeypatch.setattr(sf, "settings_fp", lambda stage: "changed" if stage == "extract" else real(stage))
    now = sf.current_fingerprints()
    assert now["listen"].effective_fp == base["listen"].effective_fp
    assert now["extract"].prompt_fp == base["extract"].prompt_fp
    assert now["extract"].effective_fp != base["extract"].effective_fp
    monkeypatch.setattr(sf, "settings_fp", lambda stage: "changed" if stage == "listen" else real(stage))
    now = sf.current_fingerprints()
    assert now["listen"].effective_fp != base["listen"].effective_fp
    assert now["extract"].effective_fp != base["extract"].effective_fp              # extract builds on listen


def test_the_plan_says_whether_the_prompt_or_the_settings_changed(monkeypatch):
    old = sf.current_fingerprints()
    stored = {s: fp.effective_fp for s, fp in old.items()}
    details = {s: {"prompt_fp": fp.prompt_fp, "settings_fp": fp.settings_fp} for s, fp in old.items()}
    real = sf.settings_fp
    monkeypatch.setattr(sf, "settings_fp", lambda stage: "x" if stage == "extract" else real(stage))
    p = sf.plan(stored, sf.current_fingerprints(), details)
    assert (p.rerun_from, p.rerun, p.reuse) == ("extract", ("extract",), ("listen",))
    assert p.note() == "Reprocessing: the API settings of stage extract changed"
    assert "API settings of 'extract' changed" in p.reason_text()
    monkeypatch.setattr(sf, "settings_fp", real)
    # a changed prompt is still reported as a prompt
    p = sf.plan(dict(stored, extract="e" * 16), old,
                dict(details, extract={"prompt_fp": "0" * 16, "settings_fp": details["extract"]["settings_fp"]}))
    assert p.note() == "Reprocessing: the prompt of stage extract changed" and "prompt of 'extract' changed" in p.reason_text()
    # a row from before settings were recorded counts as a change of the settings, once
    p = sf.plan(dict(stored, extract="e" * 16), old,
                dict(details, extract={"prompt_fp": details["extract"]["prompt_fp"], "settings_fp": ""}))
    assert p.note() == "Reprocessing: the API settings of stage extract changed"


# ---- the stages after the principles ----------------------------------------------------------------------------
PF = {"match": "b" * 16, "caption": "c" * 16}


def post(up="u" * 16, pf=PF):
    return sf.post_current(up, pf)


def eff_of(cur):
    return {s: fp.effective_fp for s, fp in cur.items()}


def test_nothing_is_redone_when_the_post_stages_are_current():
    cur = post()
    assert sf.post_plan(eff_of(cur), cur).redo == ()


def test_a_changed_match_prompt_redoes_match_and_then_the_caption():
    p = sf.post_plan(eff_of(post()), post(pf=dict(PF, match="z" * 16)))
    assert p.redo == ("match", "caption")


def test_a_changed_caption_prompt_redoes_only_the_caption():
    assert sf.post_plan(eff_of(post()), post(pf=dict(PF, caption="z" * 16))).redo == ("caption",)


def test_a_changed_principle_stage_redoes_both():
    assert sf.post_plan(eff_of(post()), post(up="v" * 16)).redo == ("match", "caption")


def test_a_post_stage_with_no_record_is_redone_once():
    p = sf.post_plan({}, post())
    assert p.redo == ("match", "caption") and set(p.kinds.values()) == {"missing"}


def test_a_changed_setting_of_a_post_stage_redoes_only_that_stage(monkeypatch):
    old = post()
    details = {s: {"prompt_fp": fp.prompt_fp, "settings_fp": fp.settings_fp} for s, fp in old.items()}
    real = sf.settings_fp
    monkeypatch.setattr(sf, "settings_fp", lambda stage: "x" if stage == "caption" else real(stage))
    p = sf.post_plan(eff_of(old), post(), details)
    assert p.redo == ("caption",) and p.kinds == {"caption": "settings"}
    assert "caption: API settings changed" in p.reason_text()


# ---- storage ----------------------------------------------------------------------------------------------------
def test_a_database_at_schema_v4_gets_the_settings_column(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    store.count_all()
    con = sqlite3.connect(store.DB_PATH)
    con.execute("ALTER TABLE stage_fingerprints DROP COLUMN settings_fp")
    con.execute("PRAGMA user_version = 4")
    con.commit()
    con.close()
    store._prepared.clear()
    store.save_stage_fingerprint("v", "listen", "p", "e", settings_fp="s")
    assert store.get_stage_fingerprints("v")["listen"]["settings_fp"] == "s"
    assert rows("PRAGMA user_version") == [(8,)]
    store._prepared.clear()


def test_archiving_the_post_stage_results_moves_them_and_deletes_nothing(folders, pipe):
    pid, iid, _ = seed_video(folders)
    store.set_image_caption(iid, "كابشن")
    assert store.archive_post_results(VID, ("match", "caption"), "why") == {"image_links": 1, "captions": 1}
    assert rows("SELECT * FROM principle_images") == []
    assert get_image(iid).standalone_caption == ""
    assert rows("SELECT principle_id, image_id FROM principle_images_archive") == [(pid, iid)]
    assert rows("SELECT image_id, standalone_caption, reason FROM images_history") == [(iid, "كابشن", "why")]
    assert [p.principle for p in store.get_principles_for_video(VID)] == ["مبدأ"]      # principles stay


# ---- the bookmarklet path ---------------------------------------------------------------------------------------
def test_a_changed_match_prompt_matches_again_and_captions_again_without_touching_the_principles(folders, pipe):
    pid, iid, _ = seed_video(folders)
    new_prompts = dict(enrichment.post_stage_prompt_fps(), match="9" * 16)               # the match prompt was edited
    pipe.monkeypatch.setattr(pipeline, "post_stage_prompt_fps", lambda: new_prompts)
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", lambda principles, images, log=print: {0: [0]})
    res = pipeline.process_video(URL, log=quiet)
    assert res.ok and "match, caption redone" in res.message
    assert [im.id for im in store.get_images_for_principle(pid)] == [iid]            # matched again
    assert rows("SELECT principle_id, image_id FROM principle_images_archive") == [(pid, iid)]
    assert [p.principle for p in store.get_principles_for_video(VID)] == ["مبدأ"]
    assert rows("SELECT * FROM principles_archive") == [] and pipe.sent == []          # no principle redone, nothing sent
    fps = store.get_stage_fingerprints(VID)
    assert fps["match"]["prompt_fp"] == "9" * 16
    assert fps["match"]["settings_fp"] == llm.settings_fp("match")
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."


def test_a_changed_caption_prompt_captions_the_unmatched_images_again_and_keeps_the_old_caption(folders, pipe):
    pid, iid, _ = seed_video(folders)
    con = sqlite3.connect(store.DB_PATH)
    con.execute("DELETE FROM principle_images WHERE image_id = ?", (iid,))
    con.commit()
    con.close()
    store.set_image_caption(iid, "كابشن قديم")
    pipeline.record_post_fingerprints(VID)
    store.save_stage_fingerprint(VID, "caption", "0" * 16, "f" * 16)
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: pytest.fail("matched again"))
    pipe.monkeypatch.setattr(pipeline, "caption_standalone_image", lambda image, transcript, log=print: "كابشن جديد")
    res = pipeline.process_video(URL, log=quiet)
    assert res.ok and "caption redone" in res.message
    assert get_image(iid).standalone_caption == "كابشن جديد"
    assert rows("SELECT standalone_caption FROM images_history") == [("كابشن قديم",)]
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."


def test_a_changed_setting_of_a_post_stage_redoes_it_and_records_the_new_settings(folders, pipe):
    pid, iid, _ = seed_video(folders)
    real = sf.settings_fp
    pipe.monkeypatch.setattr(sf, "settings_fp", lambda stage: "changed" if stage == "caption" else real(stage))
    res = pipeline.process_video(URL, log=quiet)
    assert res.ok and "caption redone (caption: API settings changed)" in res.message
    assert store.get_stage_fingerprints(VID)["caption"]["settings_fp"] == "changed"
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."


def test_a_changed_setting_of_listen_redoes_every_stage_and_tells_the_user_why(env, monkeypatch):
    old, _, _ = seed()
    real = sf.settings_fp
    monkeypatch.setattr(sf, "settings_fp", lambda stage: "changed" if stage == "listen" else real(stage))
    listened = []

    def counting(audio, prompt, system, context):
        listened.append(context)
        return heard(audio, prompt, system, context)

    llm.set_test_backend(audio=counting, json=extract_ok)
    res = pipeline.process_video(URL, log=quiet)
    assert res.ok and res.added == 1 and len(listened) == 1                          # listened again, then extracted again
    assert rows("SELECT id FROM principles_archive") == [(old,)]
    assert any("Reprocessing: the API settings of stage listen changed" in text for text in env)
    fps = store.get_stage_fingerprints(VID)
    assert fps["listen"]["settings_fp"] == "changed" and fps["extract"]["effective_fp"] == sf.current_fingerprints()["extract"].effective_fp
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."
