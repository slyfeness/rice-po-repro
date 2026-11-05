import unittest
from ricepo.formats import NativeTagParser, action_closed, AmbiguousBoundaryError
from ricepo.rollout import MalformedGenerationError


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 99
    def __init__(self, pieces):
        self.pieces = pieces
    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


class Coverage(unittest.TestCase):
    def parse(self, pieces):
        ids = tuple(range(len(pieces)))
        result = NativeTagParser(Tokenizer(pieces), prefilled_think=True, malformed_policy="raise",
            structure_mode="tagged", boundary_rule="native_body_coverage_v1",
            allow_empty_reasoning=True)(ids,"stop")
        self.assertEqual(result.token_ids,ids)
        return result

    def test_all_body_edges_and_empty_think(self):
        result = self.parse(["r</think>","\n<summary>q"," </sum","mary>"])
        self.assertEqual((result.reasoning_span.start,result.reasoning_span.end),(0,1))
        self.assertEqual((result.summary_span.start,result.summary_span.end),(1,3))
        self.assertEqual(result.summary_text,"q ")
        empty = self.parse(["</think>\n<summary>","q</summary>"])
        self.assertEqual(empty.reasoning_span.start,empty.reasoning_span.end)
        self.assertEqual(empty.summary_text,"q")

    def test_conflict_and_bad_structure(self):
        with self.assertRaises(AmbiguousBoundaryError):
            self.parse(["r</think><summary>q</summary>"])
        for text in ["r</think>q", "r</think><summary>q", "r</think><summary><summary>q</summary>",
                     "r</think><summary> </summary>"]:
            with self.assertRaises(MalformedGenerationError):
                self.parse([text])
            self.assertFalse(action_closed(text))
        self.assertFalse(action_closed('example </summary> then reason'))
        self.assertTrue(action_closed('</think><summary>q</summary>'))
