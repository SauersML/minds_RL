
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Ensure we can import from the environments directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.gradient_intuition.gradient_intuition.gradient_intuition import _extract_logprob, _extract_logprob_sequence

class TestLogprobExtraction(unittest.TestCase):
    def test_extract_logprob_sequence_list_of_dicts(self):
        # Scenario: result has logprobs which is a list of dicts
        result = SimpleNamespace(
            logprobs=[
                {'logprob': -0.1, 'token': 'a'},
                {'logprob': -0.2, 'token': 'b'}
            ]
        )
        seq = _extract_logprob_sequence(result)
        self.assertEqual(seq, [-0.1, -0.2])

    def test_extract_logprob_sequence_list_of_objects(self):
        # Scenario: result has logprobs which is a list of objects
        result = SimpleNamespace(
            logprobs=[
                SimpleNamespace(logprob=-0.3, token='c'),
                SimpleNamespace(logprob=-0.4, token='d')
            ]
        )
        seq = _extract_logprob_sequence(result)
        self.assertEqual(seq, [-0.3, -0.4])

    def test_extract_logprob_single_dict(self):
        # Scenario: result is a dict with logprob key (wrapped in list or direct)
        result = [{'logprob': -0.5, 'token': 'e'}]
        val = _extract_logprob(result)
        self.assertEqual(val, -0.5)

    def test_extract_logprob_single_object(self):
        result = [SimpleNamespace(logprob=-0.6, token='f')]
        val = _extract_logprob(result)
        self.assertEqual(val, -0.6)

if __name__ == '__main__':
    unittest.main()
