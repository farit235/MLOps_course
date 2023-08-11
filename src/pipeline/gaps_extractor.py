import random
import typing as t

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from typing_extensions import TypedDict

from .group_tasks import extract_fully_connected_component_min3, init_graph
from .word_metadata import SentenceWithGap, WordMetadataGenerator


class FormWithPosition(TypedDict):
    word: str
    sent_idx: int


class FormattedTask(TypedDict):
    options: t.List[FormWithPosition]
    sents: t.List[str]


class TaskExtractor:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None):
        tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.mask_token = tokenizer.mask_token
        self.pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        self.metadata_generator = WordMetadataGenerator()

    def get_reconstructable_spans(self, text: str) -> t.List[SentenceWithGap]:
        all_spans: t.List[SentenceWithGap] = self.metadata_generator.process_text(text)
        filtered_spans = []
        for span_data in all_spans:
            masked_sentence = span_data.sentence_with_filler.replace(
                span_data.GAP_FILLER, self.mask_token
            )
            correct_answer = span_data.form.lower()
            model_answers = [
                i["token_str"].lower() for i in self.pipeline(masked_sentence)
            ]
            if correct_answer not in model_answers:
                continue
            other_options = [i for i in model_answers if i != correct_answer]
            span_data.set_other_options_info(other_options)
            filtered_spans.append(span_data)
        return filtered_spans

    def get_several_tasks(
        self, text: str, max_tasks: int = 4, max_sents_in_task: int = 5
    ) -> t.List[FormattedTask]:
        gaps = self.get_reconstructable_spans(text)
        graph = init_graph(gaps)
        tasks: t.List[t.List[SentenceWithGap]] = []

        for i in range(max_tasks):
            current_task_sent_ids = extract_fully_connected_component_min3(graph)
            if not current_task_sent_ids:
                break
            current_task = [gaps[i] for i in current_task_sent_ids]
            if len(current_task) > max_sents_in_task:
                current_task = random.choices(current_task, k=max_sents_in_task)
            tasks.append(current_task)
            current_task_sents = set([gap.original_sentence for gap in current_task])

            nodes_to_remove = []
            for node in graph.nodes():
                if gaps[node].original_sentence in current_task_sents:
                    nodes_to_remove.append(node)
            for node in nodes_to_remove:
                graph.remove_node(node)

        return [self._task_to_json(task) for task in tasks]

    def _task_to_json(self, task: t.List[SentenceWithGap]) -> FormattedTask:
        forms: t.List[FormWithPosition] = []
        for idx, item in enumerate(task):
            option = FormWithPosition(
                {
                    "word": item.sentence_with_filler,
                    "sent_idx": idx,
                }
            )
            forms.append(option)

        sents = [item.sentence_with_filler for item in task]
        random.shuffle(forms)
        formatted_task = FormattedTask({"options": forms, "sents": sents})
        return formatted_task
