# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike


@pytest.fixture(scope="module")
def default_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")

# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the GLM tool-parser tests."""
import os

import pytest

from vllm.tokenizers import get_tokenizer

MODEL = "zai-org/GLM-4.7"


@pytest.fixture
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=os.environ.get("VLLM_TEST_GLM_TOKENIZER", MODEL))


@pytest.fixture
def glm53_tool_profile(monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CALLING_PROFILE", "glm53")
