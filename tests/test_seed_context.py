import json
import tempfile
import unittest
from pathlib import Path

from research_agent.errors import ExtractionError, StructuredOutputError
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_mineru_markdown,
)
from research_agent.llm.fake import FakeLLMClient
from research_agent.models.seed_extraction import SeedContextExtractionDraft
from research_agent.modules.seed_context import (
    SeedContextConfig,
    SeedContextExtractor,
    write_seed_context_outputs,
)
from research_agent.serialization import loads_json


SAMPLE_MARKDOWN = """# Test Paper: A Deterministic Title

Author information

## ABSTRACT
This abstract states the paper objective and reported result.

## 1 INTRODUCTION
The introduction motivates the problem and mentions a citation-only RefBaseline.

## 2 RELATED WORK
Related Work remains part of the scientific argument and must be supplied.

## 3 METHOD
The method uses an explicit mechanism.

<table>
<tr><td>Dataset-A</td><td>0.9</td></tr>
</table>

$$
x_{t+1} = f(x_t)
$$

![image](https://example.invalid/figure.png)

Figure 1: A method illustration.

## 4 EXPERIMENTS
We evaluate Dataset-A with Accuracy and compare against Baseline-A.

## 5 DISCUSSION
The authors note a method trade-off and state that future work should extend
the mechanism to cover a broader operating condition.

## REFERENCES
RefBaseline and RefDataset must never enter the extraction context.

## A IMPLEMENTATION DETAILS
Additional implementation settings are in this appendix.
"""


def _draft(section_id: str, *, paper_id: str | None = "test-paper") -> dict:
    evidence = {"value": "A supported summary.", "source_section_ids": [section_id]}
    item = {"value": "Supported item.", "source_section_ids": [section_id]}
    return {
        "paper_id": paper_id,
        "title": "LLM must not replace the deterministic title",
        "abstract": "LLM must not replace the deterministic abstract",
        "research_objective": evidence,
        "core_method": {"value": "The supported method.", "source_section_ids": [section_id]},
        "main_components": [item],
        "main_findings": [
            {"value": "The reported finding.", "source_section_ids": [section_id]}
        ],
        "limitations": [
            {"value": "The stated limitation.", "source_section_ids": [section_id]}
        ],
        "experiment_context": {
            "tasks_or_datasets": [
                {"value": "Dataset-A", "source_section_ids": [section_id]}
            ],
            "metrics": [{"value": "Accuracy", "source_section_ids": [section_id]}],
            "major_baselines": [
                {"value": "Baseline-A", "source_section_ids": [section_id]}
            ],
            "experimental_setting": {
                "value": "Controlled evaluation.",
                "source_section_ids": [section_id],
            },
        },
    }


class SeedContextTests(unittest.TestCase):
    def test_parser_builds_full_body_and_preserves_blocks(self) -> None:
        parsed = parse_mineru_markdown(SAMPLE_MARKDOWN, paper_id="test-paper")
        reading = build_full_paper_reading_context(parsed)

        self.assertEqual(parsed.title, "Test Paper: A Deterministic Title")
        self.assertEqual(
            parsed.abstract,
            "This abstract states the paper objective and reported result.",
        )
        self.assertIsNotNone(parsed.references_start_line)
        self.assertIsNone(parsed.appendix_start_line)
        self.assertEqual(len(parsed.html_tables), 1)
        self.assertIn("Dataset-A", parsed.html_tables[0].text)
        self.assertEqual(len(parsed.equation_blocks), 1)
        self.assertIn("x_{t+1}", parsed.equation_blocks[0].text)
        self.assertEqual(len(parsed.image_placeholders), 1)
        self.assertIn("figure.png", parsed.image_placeholders[0].text)

        headings = [section.heading for section in reading.body_sections]
        self.assertIn("2 RELATED WORK", headings)
        self.assertIn("3 METHOD", headings)
        self.assertIn("4 EXPERIMENTS", headings)
        rendered = reading.render()
        self.assertIn("Related Work remains part", rendered)
        self.assertIn("Dataset-A", rendered)
        self.assertNotIn("RefDataset must never enter", rendered)
        self.assertNotIn("Additional implementation settings", rendered)
        self.assertIn("FULL MAIN-PAPER BODY", rendered)

    def test_references_are_the_only_tail_boundary(self) -> None:
        text = """# Paper

## ABSTRACT
Abstract.

## 1 INTRODUCTION
Body.

## APPENDIX
Appendix first.

## REFERENCES
References second.
"""
        parsed = parse_mineru_markdown(text, "boundary-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertEqual(reading.cut_line, parsed.references_start_line)
        self.assertIsNone(parsed.appendix_start_line)
        self.assertIn("Appendix first", reading.body_text)
        self.assertNotIn("References second", reading.body_text)

    def test_single_letter_headings_do_not_trigger_appendix_cutoff(self) -> None:
        text = """# A General Framework for Reasoning

## ABSTRACT
Abstract.

## 1 INTRODUCTION
Main introduction.

## C Multi-view Reasoning
This is a main-paper scientific section.
"""
        parsed = parse_mineru_markdown(text, "letter-heading-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertIsNone(parsed.appendix_start_line)
        self.assertEqual(reading.cut_line, None)
        self.assertIn("A General Framework for Reasoning", reading.body_text)
        self.assertIn("C Multi-view Reasoning", reading.body_text)

    def test_appendix_and_supplementary_headings_do_not_cut_body(self) -> None:
        text = """# Paper

## ABSTRACT
Abstract.

## METHOD
Main method.

## Supplementary Materials
Supplementary details.
"""
        parsed = parse_mineru_markdown(text, "supplementary-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertIsNone(parsed.appendix_start_line)
        self.assertIsNone(reading.cut_line)
        self.assertIn("Main method", reading.body_text)
        self.assertIn("Supplementary details", reading.body_text)

    def test_no_references_or_appendix_preserves_complete_body(self) -> None:
        text = """# Paper

## ABSTRACT
Abstract.

## METHOD
Complete main-paper method.
"""
        parsed = parse_mineru_markdown(text, "complete-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertIsNone(parsed.references_start_line)
        self.assertIsNone(parsed.appendix_start_line)
        self.assertIsNone(reading.cut_line)
        self.assertIn("Complete main-paper method", reading.body_text)

    def test_reference_heading_variants_are_exact_tail_boundaries(self) -> None:
        for heading in ("Reference", "References", "REFERENCES", "Bibliography"):
            text = f"""# Paper

## METHOD
Main body.

## {heading}
Tail material.

## Appendix
More tail material.
"""
            with self.subTest(heading=heading):
                parsed = parse_mineru_markdown(text, "reference-variant-paper")
                reading = build_full_paper_reading_context(parsed)
                self.assertIsNotNone(parsed.references_start_line)
                self.assertEqual(reading.cut_line, parsed.references_start_line)
                self.assertIn("Main body", reading.body_text)
                self.assertNotIn("Tail material", reading.body_text)
                self.assertNotIn("More tail material", reading.body_text)

    def test_reference_like_body_heading_is_not_a_tail_boundary(self) -> None:
        text = """# Paper

## METHOD
Main body.

## References to Prior Work
This is still part of the scientific discussion.

## Reference-based Analysis
This is also part of the main paper.
"""
        parsed = parse_mineru_markdown(text, "reference-like-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertIsNone(parsed.references_start_line)
        self.assertIsNone(reading.cut_line)
        self.assertIn("References to Prior Work", reading.body_text)
        self.assertIn("Reference-based Analysis", reading.body_text)

    def test_mineru_pseudocode_heading_stays_in_surrounding_section(self) -> None:
        text = "# Paper\n\n## ABSTRACT\nAbstract.\n\n## 1 INTRODUCTION\n# param p, update u from OPT\nThe artifact remains text.\n\n## 2 METHOD\nMethod."
        parsed = parse_mineru_markdown(text, "pseudo-paper")
        reading = build_full_paper_reading_context(parsed)
        self.assertFalse(any(section.heading.startswith("param ") for section in parsed.sections))
        self.assertIn("# param p, update u from OPT", reading.render())

    def test_one_call_contains_all_main_sections_and_no_tail(self) -> None:
        parsed = parse_mineru_markdown(SAMPLE_MARKDOWN, paper_id="test-paper")
        method_id = next(
            section.section_id
            for section in parsed.sections
            if section.heading == "3 METHOD"
        )
        client = FakeLLMClient(responses=[_draft(method_id, paper_id="wrong-paper")])

        result = SeedContextExtractor(client).extract_text(SAMPLE_MARKDOWN, "test-paper")

        self.assertEqual(len(client.calls), 1)
        request_text = client.calls[0].messages[1].content
        self.assertIn("2 RELATED WORK", request_text)
        self.assertIn("3 METHOD", request_text)
        self.assertIn("4 EXPERIMENTS", request_text)
        self.assertIn("5 DISCUSSION", request_text)
        self.assertIn("future work should extend", request_text)
        self.assertNotIn("RefDataset must never enter", request_text)
        self.assertNotIn("Additional implementation settings", request_text)
        self.assertEqual(result.context.paper_id, "test-paper")
        self.assertEqual(result.context.title, parsed.title)
        self.assertEqual(result.context.abstract, parsed.abstract)
        self.assertEqual(result.context.research_objective, "A supported summary.")
        self.assertIn("llm_paper_id_ignored", result.audit.warnings)
        self.assertEqual(len(result.audit.llm_call_metadata), 1)
        self.assertEqual(
            result.audit.llm_call_metadata[0].input_chars,
            len(build_full_paper_reading_context(parsed).render()),
        )
        self.assertTrue(any(anchor.field_path == "core_method" for anchor in result.audit.anchors))
        self.assertNotIn("anchors", result.context.model_dump())

    def test_author_future_direction_from_discussion_is_preserved_as_limitation(self) -> None:
        parsed = parse_mineru_markdown(SAMPLE_MARKDOWN, paper_id="test-paper")
        discussion_id = next(
            section.section_id
            for section in parsed.sections
            if section.heading == "5 DISCUSSION"
        )
        response = _draft(discussion_id)
        response["limitations"] = [
            {
                "value": "The authors state that future work should extend the mechanism to a broader operating condition.",
                "source_section_ids": [discussion_id],
            }
        ]
        result = SeedContextExtractor(
            FakeLLMClient(responses=[response])
        ).extract_text(SAMPLE_MARKDOWN, "test-paper")
        self.assertEqual(
            result.context.limitations,
            [
                "The authors state that future work should extend the mechanism to a broader operating condition."
            ],
        )
        self.assertTrue(
            any(
                anchor.field_path == "limitations[0]"
                and anchor.section_id == discussion_id
                for anchor in result.audit.anchors
            )
        )

    def test_long_main_body_still_uses_one_call_without_truncation(self) -> None:
        long_marker = "END-OF-MAIN-BODY-" + "z" * 100
        long_text = "\n\n".join(
            [
                "# Long Paper",
                "## ABSTRACT\nShort abstract.",
                "## 1 INTRODUCTION\nINTRO " + "a" * 20_000,
                "## 2 RELATED WORK\nRELATED " + "r" * 20_000,
                "## 3 METHOD\nMETHOD " + "b" * 20_000,
                "## 4 EXPERIMENTS\nEXPERIMENT " + "c" * 20_000,
                long_marker,
                "## REFERENCES\nReference tail.",
            ]
        )
        parsed = parse_mineru_markdown(long_text, paper_id="long-paper")
        method_id = next(section.section_id for section in parsed.sections if section.heading == "3 METHOD")
        client = FakeLLMClient(responses=[_draft(method_id, paper_id="long-paper")])

        result = SeedContextExtractor(
            client, SeedContextConfig(max_tokens=500)
        ).extract_text(long_text, "long-paper")

        self.assertEqual(len(client.calls), 1)
        request_text = client.calls[0].messages[1].content
        self.assertIn(long_marker, request_text)
        self.assertNotIn("Reference tail", request_text)
        self.assertEqual(
            result.audit.llm_call_metadata[0].input_chars,
            len(build_full_paper_reading_context(parsed).render()),
        )

    def test_tail_section_ids_cannot_be_used_as_evidence(self) -> None:
        parsed = parse_mineru_markdown(SAMPLE_MARKDOWN, paper_id="test-paper")
        tail_ids = [
            section.section_id
            for section in parsed.sections
            if section.is_references or section.is_appendix
        ]
        self.assertTrue(tail_ids)
        with self.assertRaises(ExtractionError):
            SeedContextExtractor(
                FakeLLMClient(responses=[_draft(tail_ids[0])])
            ).extract_text(SAMPLE_MARKDOWN, "test-paper")

    def test_deterministic_identity_and_missing_fields(self) -> None:
        client = FakeLLMClient(responses=[{"paper_id": "wrong-paper", "title": "wrong"}])
        result = SeedContextExtractor(client).extract_text(SAMPLE_MARKDOWN, "test-paper")
        self.assertEqual(result.context.paper_id, "test-paper")
        self.assertEqual(result.context.title, "Test Paper: A Deterministic Title")
        self.assertEqual(
            result.context.abstract,
            "This abstract states the paper objective and reported result.",
        )
        self.assertIsNone(result.context.research_objective)
        self.assertEqual(result.context.main_components, [])
        self.assertEqual(result.context.experiment_context.metrics, [])

    def test_invalid_source_id_and_missing_evidence_fail(self) -> None:
        invalid_id = _draft("S999")
        with self.assertRaises(ExtractionError):
            SeedContextExtractor(FakeLLMClient(responses=[invalid_id])).extract_text(
                SAMPLE_MARKDOWN, "test-paper"
            )

        no_evidence = _draft("S3")
        no_evidence["research_objective"] = {
            "value": "unsupported",
            "source_section_ids": [],
        }
        with self.assertRaises(StructuredOutputError):
            SeedContextExtractor(FakeLLMClient(responses=[no_evidence])).extract_text(
                SAMPLE_MARKDOWN, "test-paper"
            )

    def test_extra_field_is_rejected_by_draft_schema(self) -> None:
        response = _draft("S4")
        response["unexpected"] = "not allowed"
        with self.assertRaises(StructuredOutputError):
            SeedContextExtractor(FakeLLMClient(responses=[response])).extract_text(
                SAMPLE_MARKDOWN, "test-paper"
            )

    def test_prompt_contains_canonical_json_schema_and_rules(self) -> None:
        client = FakeLLMClient(responses=[{"paper_id": "test-paper"}])
        SeedContextExtractor(client).extract_text(SAMPLE_MARKDOWN, "test-paper")
        system_prompt = client.calls[0].messages[0].content
        schema = json.dumps(
            SeedContextExtractionDraft.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertIn(schema, system_prompt)
        self.assertIn("JSON", system_prompt)
        self.assertIn("source_section_ids", system_prompt)
        self.assertIn("experiment_context", system_prompt)
        self.assertIn("outside knowledge", system_prompt)
        self.assertIn("Return `null` for `paper_id`, `title`, and `abstract`", system_prompt)
        self.assertIn("do not copy long passages", system_prompt)
        self.assertIn("For `limitations`, collect author-supported statements", system_prompt)
        self.assertIn("future work", system_prompt)
        self.assertIn("Discussion, or Conclusion", system_prompt)
        self.assertIn("Do not perform Gap Discovery", system_prompt)
        self.assertNotIn("{{SEED_CONTEXT_EXTRACTION_DRAFT_JSON_SCHEMA}}", system_prompt)

    def test_output_files_use_separate_context_and_audit_envelopes(self) -> None:
        result = SeedContextExtractor(
            FakeLLMClient(responses=[_draft("S4")])
        ).extract_text(SAMPLE_MARKDOWN, "test-paper")
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path, audit_path = write_seed_context_outputs(result, temp_dir)
            context = loads_json(context_path.read_text(encoding="utf-8"), "seed_paper_context")
            audit = loads_json(audit_path.read_text(encoding="utf-8"), "seed_extraction_audit")
        self.assertEqual(context["payload"]["paper_id"], "test-paper")
        self.assertNotIn("anchors", context["payload"])
        self.assertIn("anchors", audit["payload"])

    def test_data_files_are_not_an_extraction_output_target(self) -> None:
        data_dir = Path(__file__).parents[1] / "data"
        markdown_files = list(data_dir.rglob("*.md"))
        self.assertGreater(len(markdown_files), 0)
        self.assertTrue(all(path.is_file() for path in markdown_files))
        self.assertTrue(
            all("seed_context.json" not in path.name for path in markdown_files)
        )


if __name__ == "__main__":
    unittest.main()
