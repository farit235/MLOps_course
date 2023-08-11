import typing as t

from networkx import Graph

from .word_metadata import SentenceWithGap


def may_be_used_in_one_task(sent1: SentenceWithGap, sent2: SentenceWithGap) -> bool:
    return (
        # a task should focus on one part of speech - this rule is commented to minimize number of empty tasks during prediction
        sent1.pos == sent2.pos
        # do not use same words or same original sentences in one task
        and sent1.form != sent2.form
        and sent1.stem != sent2.stem
        and sent1.original_sentence != sent2.original_sentence
        # word should not suit several gaps (according to model prediction)
        and sent1.form not in sent2.other_options
        and sent2.form not in sent1.other_options
    )


def init_graph(sentences_with_spans: t.List[SentenceWithGap]) -> Graph:
    graph = Graph()

    for i, sent1 in enumerate(sentences_with_spans):
        for j, sent2 in enumerate(sentences_with_spans):
            if i > j and may_be_used_in_one_task(sent1, sent2):
                graph.add_edge(i, j)
    return graph


# toDo implement algorithm for larger minimal number of tasks
def extract_fully_connected_component_min3(graph: Graph) -> t.Optional[t.List[int]]:
    for start_node in graph.nodes():
        start_node_neighbors = set(graph.neighbors(start_node))

        for neighbor in start_node_neighbors:
            current_neighbor_neighbors = [
                i for i in list(graph.neighbors(neighbor)) if i != start_node
            ]
            current_component = [start_node, neighbor]
            for step3_node in current_neighbor_neighbors:
                step3_node_neighbors = graph.neighbors(step3_node)
                if all(
                    [
                        already_selected in step3_node_neighbors
                        for already_selected in current_component
                    ]
                ):
                    current_component.append(step3_node)
            if len(current_component) >= 3:
                return current_component
    return None
