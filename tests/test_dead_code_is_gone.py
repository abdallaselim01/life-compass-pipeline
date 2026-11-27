"""Nothing dead stays (design decision 2026-11-13): the code no entry point reaches is gone, and the stage names have a guard."""
import pytest

from lifecompass import audio, enrichment, llm, pipeline, resurface, store
from lifecompass import stage_fingerprints as sf


def test_the_dead_code_is_gone():
    gone = ((audio, ("cleanup",)),
            (enrichment, ("find_related", "LINK_PROMPT_TEMPLATE", "LinkResult", "find_related_many", "LINK_MANY_PROMPT_TEMPLATE",
                          "LinkItem", "ManyLinkEntry", "ManyLinkResult", "RELATED_BATCH", "RELATED_NEW_PER_CALL")),
            (llm, ("generate_text", "_TEST_TEXT")), (resurface, ("SHOW_GENERATED_CONTEXT", "SHOW_RELATED_LIST")),
            (pipeline, ("retry_links",)),
            (store, ("get_related_with_notes", "delete_document", "get_image", "link_principles", "all_principles", "get_related", "get_principle")))
    for module, names in gone:
        for name in names:
            assert not hasattr(module, name), f"{module.__name__}.{name} is back"
    assert not hasattr(store.Principle(id=None, principle="x"), "related_ids")


def test_the_links_stage_is_gone_from_every_stage_list():
    assert sf.POST_STAGES == ("match", "caption")
    assert set(enrichment.post_stage_prompt_fps()) == {"match", "caption"}
    assert "links" not in llm.STAGES and "links" not in llm._JSON_STAGES and "find_related" not in llm._LONG_WAIT_KINDS
    with pytest.raises(ValueError, match="unknown stage"):
        llm.request_settings("links")


def test_a_misspelt_stage_name_is_refused_instead_of_quietly_getting_generic_settings():
    for function in (llm.request_settings, llm.settings_fp, llm.max_quality_config_kwargs):
        with pytest.raises(ValueError, match="unknown stage"):
            function("matchh")
    assert llm.request_settings(None)                                          # no stage = the generic settings, as before
    assert all(llm.request_settings(stage) for stage in llm.STAGES)
