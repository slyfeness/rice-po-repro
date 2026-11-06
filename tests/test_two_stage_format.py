import unittest
from ricepo.two_stage import TwoStageFormat
from ricepo.contracts import Generation,InvalidGeneration,generation_action_mask
from test_native_byte_coverage import BytesTokenizer

class Tok(BytesTokenizer):
    all_special_tokens=['<eos>']
    def encode(self,text,add_special_tokens=False):return list(text.encode('utf8'))
    def convert_ids_to_tokens(self,ids):return ['<eos>' if t==1000 else self.alphabet[t] for t in ids]

class TwoStageFormatTests(unittest.TestCase):
    def test_eos_summary_preserves_text_ids_and_environment_ownership(self):
        f=TwoStageFormat(Tok());a=tuple('reason正</think>'.encode());b=tuple(' exact 正 '.encode())+(1000,)
        g=f.record(a,b,(-1.,)*len(a),(-2.,)*len(b),'think_closed','eos')
        self.assertIsInstance(g,Generation)
        self.assertEqual(g.token_ids,a+f.control_ids+b)
        self.assertEqual(g.summary_text,' exact 正 ')
        self.assertEqual(generation_action_mask(g),(True,)*len(a)+(False,)*len(f.control_ids)+(True,)*len(b))
        self.assertTrue(all(g.action_mask[g.summary_span.start:g.summary_span.end]))

    def test_nested_empty_and_unclosed_think_are_real_failures(self):
        f=TwoStageFormat(Tok())
        for a,b,stop1,stop2 in [(b'r<summary>x</think>',b's</summary>','think_closed','summary_closed'),
                                (b'r</think>',b'','think_closed','eos'),
                                (b'r',b'','eos','not_started')]:
            b=tuple(b)+(1000,) if stop2=='eos' else tuple(b)
            g=f.record(tuple(a),b,(-1.,)*len(a),(-1.,)*len(b),stop1,stop2)
            self.assertIsInstance(g,InvalidGeneration)
            self.assertEqual(sum(g.action_mask),len(a)+len(b))

    def test_invalid_utf8_is_retained_as_unextractable_action_not_repaired(self):
        from dataclasses import asdict
        from ricepo.contracts import generation_from_record
        f=TwoStageFormat(Tok());a=tuple(b'r</think>');b=tuple(b's\xbe</summary>')
        g=f.record(a,b,(-1.,)*len(a),(-2.,)*len(b),'think_closed','summary_closed')
        self.assertIsInstance(g,InvalidGeneration)
        self.assertEqual(g.category,'native_text_encoding')
        self.assertEqual(g.token_ids,a+f.control_ids+b)
        self.assertEqual(sum(g.action_mask),len(a)+len(b))
        self.assertEqual(generation_from_record(asdict(g)),g)

if __name__=='__main__':unittest.main()
