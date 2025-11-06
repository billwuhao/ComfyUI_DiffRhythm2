# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
from typing import List, Pattern, Union

from phonemizer.utils import list2str, str2list
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

# ✅ 新增导入
try:
    import pyopenjtalk
    HAS_PYOPENJTALK = True
except ImportError:
    HAS_PYOPENJTALK = False
    print("[WARN] pyopenjtalk not found. Japanese G2P will fallback to Espeak.")

class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="|_|", syllable="-", phone="|"),
        preserve_punctuation=True,
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:

        self.language = language
        self.preserve_punctuation_marks = ",.?!;:'…"
        self.separator = separator

        # ✅ 如果语言是日语并且 pyopenjtalk 可用，就不用 Espeak
        if self.language.startswith("ja") and HAS_PYOPENJTALK:
            self.backend = None  # 用 pyopenjtalk 手动处理
        else:
            self.backend = EspeakBackend(
                language,
                punctuation_marks=self.preserve_punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )

    # convert chinese punctuation to english punctuation
    def convert_chinese_punctuation(self, text: str) -> str:
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def __call__(self, text, strip=True) -> List[str]:
        text_type = type(text)
        normalized_text = []
        for line in str2list(text):
            line = self.convert_chinese_punctuation(line.strip())
            line = re.sub(r"[^\w\s_,\.\?!;:\'…]", "", line)
            line = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", line)
            line = re.sub(r"\s+", " ", line)
            normalized_text.append(line)

        # ✅ 如果是日语，用 pyopenjtalk.g2p
        if self.language.startswith("ja") and HAS_PYOPENJTALK:
            phonemized = [pyopenjtalk.g2p(line, kana=False) for line in normalized_text]
        else:
            phonemized = self.backend.phonemize(
                normalized_text, separator=self.separator, strip=strip, njobs=1
            )

        # 后处理：统一符号分隔逻辑
        if text_type == str:
            phonemized = re.sub(r"([,\.\?!;:\'…])", r"|\1|", list2str(phonemized))
            phonemized = re.sub(r"\|+", "|", phonemized)
            phonemized = phonemized.rstrip("|")
        else:
            for i in range(len(phonemized)):
                phonemized[i] = re.sub(r"([,\.\?!;:\'…])", r"|\1|", phonemized[i])
                phonemized[i] = re.sub(r"\|+", "|", phonemized[i])
                phonemized[i] = phonemized[i].rstrip("|")
        return phonemized
