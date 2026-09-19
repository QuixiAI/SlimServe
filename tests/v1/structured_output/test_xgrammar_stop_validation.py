from unittest.mock import Mock

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class StopAwareMatcher:
    def __init__(self, stop_token):
        self.stop_token = stop_token
        self.terminated = False
        self.accepted = []
        self.rollback_calls = []

    def accept_token(self, token):
        if self.terminated:
            raise AssertionError("advanced after stop token")
        self.accepted.append(token)
        if token == self.stop_token:
            self.terminated = True
        return True

    def is_terminated(self):
        return self.terminated

    def rollback(self, count):
        self.rollback_calls.append(count)
        del self.accepted[-count:]
        self.terminated = False


def test_validate_tokens_stops_at_grammar_stop_token():
    matcher = StopAwareMatcher(stop_token=99)
    grammar = XgrammarGrammar(vocab_size=1000, matcher=matcher, ctx=Mock())

    assert grammar.validate_tokens([1, 99, 198]) == [1, 99]
    assert matcher.accepted == []
    assert matcher.rollback_calls == [2]
    assert grammar.is_terminated() is False


def test_validate_tokens_skips_already_terminated_matcher():
    matcher = Mock()
    grammar = XgrammarGrammar(vocab_size=1000, matcher=matcher, ctx=Mock())
    grammar._is_terminated = True

    assert grammar.validate_tokens([198]) == []
    matcher.accept_token.assert_not_called()
