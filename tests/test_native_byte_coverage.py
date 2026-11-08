"""A split UTF-8 character is not a malformed model action."""
import unittest
from types import SimpleNamespace
from ricepo.formats import NativeTagParser, AmbiguousBoundaryError
from ricepo.contracts import Span


class BytesTokenizer:
    eos_token_id=1000
    pad_token_id=1000
    backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace(__getstate__=lambda:b'{"type":"ByteLevel"}'))
    def __init__(self):
        base=list(range(33,127))+list(range(161,173))+list(range(174,256))
        order=base+[b for b in range(256) if b not in base]
        chars=base+list(range(256,256+256-len(base)))
        self.alphabet=dict(zip(order,map(chr,chars)))
    def get_added_vocab(self):return {"<eos>":1000}
    def convert_ids_to_tokens(self,ids):return [self.alphabet[i] for i in ids]
    def decode(self,ids,**kwargs):return bytes(ids).decode("utf8",errors="replace")
    def encode(self,*args,**kwargs):raise AssertionError("native IDs must never be rebuilt")


class NativeBytesTests(unittest.TestCase):
    def test_split_unicode_is_exactly_covered_without_reencoding(self):
        tok=BytesTokenizer()
        text="reason正</think><summary>summary正</summary>"
        ids=tuple(text.encode("utf8"))+(1000,)
        parser=NativeTagParser(tok,prefilled_think=True,malformed_policy="raise",structure_mode="tagged",
            boundary_rule="native_body_coverage_v1",allow_empty_reasoning=True)
        result=parser(ids,"stop")
        self.assertEqual(result.token_ids,ids)
        self.assertEqual(result.summary_text,"summary正")
        left=len("reason正</think><summary>".encode("utf8"))
        self.assertEqual(result.summary_span,Span(left,left+len("summary正".encode("utf8"))))
        self.assertTrue(any("\ufffd" in tok.decode(ids[:i]) for i in range(len(ids)-1)))

    def test_nonmatching_complete_decoder_still_rejected(self):
        tok=BytesTokenizer()
        real_decode=tok.decode
        tok.decode=lambda ids,**kwargs:real_decode(ids,**kwargs).replace("reason","changed")
        parser=NativeTagParser(tok,prefilled_think=True,malformed_policy="raise",structure_mode="tagged",
            boundary_rule="native_body_coverage_v1")
        with self.assertRaises(AmbiguousBoundaryError):
            parser(tuple(b"reason</think><summary>summary</summary>"),"stop")

if __name__=="__main__":unittest.main()
