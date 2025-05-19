import logging
from enum import Enum
import re
from typing import List, Set, Union, Optional, Dict, Tuple, Any
import tiktoken
import random
import string
from itertools import chain
import json
from langchain.text_splitter import TextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Centralized tokenization utility
def tokenize_text(
    text: str,
    encoding_name: str = "cl100k_base",
    model_name: Optional[str] = None,
    allowed_special: Set[str] = {"<|endofprompt|>", "<|endoftext|>"},
    disallowed_special: Union[str, Set[str]] = "all",
) -> List[int]:
    """Tokenize text using tiktoken with specified parameters."""
    enc = (
        tiktoken.encoding_for_model(model_name)
        if model_name
        else tiktoken.get_encoding(encoding_name)
    )
    return enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)


class IntentType(Enum):
    Emotion = "Emotion"
    Knowledge = "Knowledge"


def select_language_desc(
    preferred_language: Optional[str],
    default_desc: str = "Identify the language of the provided Hint. Your response must be in the same language.",
) -> str:
    """Select response language description based on preferred_language format (native/es)."""
    if isinstance(preferred_language, str) and "/" in preferred_language:
        native, es = preferred_language.split("/", 1)
        logging.debug(f"Native language: {native}, Response language: {es}")
        return f"You must respond in {es}."
    logging.error(f"Invalid preferred_language format: {preferred_language}. Expected 'native/es'.")
    return default_desc


def calculate_upper_bound(
    model_limit: int = 4096,
    generate_limit: int = 512,
    tolerance: int = 500,
    raw: str = "",
    model_name: str = "gpt-3.5-turbo",
) -> int:
    """Calculate the maximum token budget for input text, ensuring it fits within model constraints."""
    raw_tokens = len(tokenize_text(raw, model_name=model_name))
    upper_bound = model_limit - raw_tokens - tolerance - generate_limit
    if upper_bound < 0:
        logging.warning(f"Raw content too long: {raw_tokens} tokens exceed limit {model_limit}")
        return 0
    return upper_bound


def equidistant_filter(chunks: List[str], separator: str, filtered_chunks_n: int = 6) -> List[str]:
    """Filter chunks to a fixed number, selecting evenly spaced elements."""
    if len(chunks) <= filtered_chunks_n:
        return chunks
    gap = (len(chunks) - 2) / (filtered_chunks_n - 2)
    indexes = [0, 1] + [int(gap * i) + 2 for i in range(filtered_chunks_n - 3)] + [-2, -1]
    return [chunks[i] for i in indexes if 0 <= i < len(chunks)]


def text_filter(text: str) -> str:
    """Clean text by normalizing excessive whitespace and line breaks."""
    # Combine patterns into a single regex for efficiency
    patterns = {
        r"[ \t]{3,}": lambda m: "\t" if "\t" in m.group() else " ",
        r"[\n\f\r\v]{3,}": "\n\n",
    }
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    return text


class TokenTextSplitter(TextSplitter):
    """Split text into token-based chunks with filtering."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: Set[str] = {"<|endofprompt|>", "<|endoftext|>"},
        disallowed_special: Union[str, Set[str]] = "all",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tokenizer = (
            tiktoken.encoding_for_model(model_name)
            if model_name
            else tiktoken.get_encoding(encoding_name)
        )
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        text = text_filter(text)
        input_ids = tokenize_text(
            text, allowed_special=self._allowed_special, disallowed_special=self._disallowed_special
        )
        chunks = []
        for start_idx in range(0, len(input_ids), self._chunk_size - self._chunk_overlap):
            end_idx = min(start_idx + self._chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:end_idx]
            chunk = self._tokenizer.decode(chunk_ids).strip()
            if chunk:
                cleaned_chunk = self._cut_meaningless_head_tail(chunk)
                if cleaned_chunk:
                    chunks.append(cleaned_chunk)
        logging.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _cut_meaningless_head_tail(self, text: str) -> str:
        """Remove insignificant head/tail sentences based on token length."""
        sentences = re.split(r"\. |! |\? |。|！|？|\n+ *\n+", text)
        if len(sentences) < 2:
            return text
        parts = []
        for i, sentence in enumerate(sentences):
            tokens = len(tokenize_text(sentence, allowed_special=self._allowed_special))
            chars = len(sentence)
            if (i == 0 or i == len(sentences) - 1) and (tokens < 20 and chars < 30):
                continue
            parts.append(sentence)
        result = "\n".join(parts)
        logging.debug(f"Trimmed text from {len(text)} to {len(result)} characters")
        return result


def chunk_filter(
    chunks: List[str], filter_func, filtered_chunks_n: int = 6, separator: str = "\n", spacer: str = "\n……\n……\n……\n"
) -> str:
    """Apply a filter to chunks and join them."""
    if len(chunks) <= filtered_chunks_n:
        return separator.join(chunks)
    filtered = filter_func(chunks, separator, filtered_chunks_n)
    return spacer.join(filtered)


def get_safe_content_truncate(content: str, model_name: str = "gpt-3.5-turbo", max_tokens: int = 3300) -> str:
    """Truncate content to fit within token limit."""
    tokens = tokenize_text(content, model_name=model_name)
    if len(tokens) <= max_tokens:
        return content
    truncated = tiktoken.encoding_for_model(model_name).decode(tokens[:max_tokens])
    logging.warning(f"Truncated content from {len(tokens)} to {max_tokens} tokens")
    return truncated


class DataType(Enum):
    DOCUMENT = "DOCUMENT"
    WEBSITE = "WEBSITE"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    AUDIO = "AUDIO"
    TEXT = "TEXT"

    @staticmethod
    def extra_values_map() -> Dict[str, str]:
        return {"SHORT_AUDIO": "AUDIO"}

    @classmethod
    def _missing_(cls, value: Any) -> "DataType":
        extra_map = cls.extra_values_map()
        mapped_value = extra_map.get(value, value)
        try:
            return cls(mapped_value)
        except ValueError:
            logging.error(f"Unknown DataType: {value}, defaulting to DOCUMENT")
            return cls.DOCUMENT


def get_urls(text: str) -> List[str]:
    """Extract unique URLs from text, sorted by length (longest first)."""
    if not text:
        return []
    pattern = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;\u4e00-\u9fa5]+[-A-Za-z0-9+&@#/%=~_|]"
    urls = re.findall(pattern, text)
    return sorted(set(urls), key=len, reverse=True)


def get_random_string(length: int) -> str:
    """Generate a random alphanumeric string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def encode_urls(text: str, random_string_len: int = 16) -> Tuple[str, Dict[str, str]]:
    """Replace URLs with random strings and return mapping."""
    urls = get_urls(text)
    random_strings = [get_random_string(random_string_len) for _ in range(len(urls))]
    url_map = dict(zip(urls, random_strings))
    for url, rand_str in url_map.items():
        text = text.replace(url, rand_str)
    return text, {rand_str: url for url, rand_str in url_map.items()}


def decode_urls(text: str, string2url_dict: Dict[str, str]) -> str:
    """Restore URLs from random strings."""
    for rand_str, url in string2url_dict.items():
        text = text.replace(rand_str, url)
    return text


class TokenParagraphSplitter(TextSplitter):
    """Split text into paragraphs and chunks, preserving sentence integrity."""

    LINE_BREAKS = "\n\f\r\v"
    WHITESPACE = " \t"
    SENTENCE_ENDS = ".!?。！？……" + LINE_BREAKS
    PAIRED_PUNCT = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("“", "”"), ("‘", "’"), ("《", "》"), ("【", "】")]
    INTRA_SENTENCE = ",，;；" + WHITESPACE

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        allowed_special: Set[str] = {"<|endofprompt|>", "<|endoftext|>"},
        disallowed_special: Union[str, Set[str]] = "all",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        """Split text into meaningful chunks."""
        text = text_filter(text)
        text, url_map = encode_urls(text)
        paragraphs = self._split_to_paragraphs(text)
        chunks = list(chain.from_iterable(self._split_to_chunks(p) for p in paragraphs))
        return [decode_urls(chunk, url_map) for chunk in chunks]

    def _split_to_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on double line breaks."""
        pattern = f"[{self.LINE_BREAKS}]+[{self.WHITESPACE}]*[{self.LINE_BREAKS}]+"
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_to_chunks(self, text: str) -> List[str]:
        """Split paragraph into token-sized chunks."""
        sentences = self._split_to_sentences(text)
        return self._merge_sentences_into_chunks(sentences)

    def _split_to_sentences(self, text: str) -> List[str]:
        """Split text into sentences, fixing fragments."""
        pattern = f"({'|'.join(map(re.escape, self.SENTENCE_ENDS))})"
        parts = re.split(pattern, text)
        sentences = ["".join(parts[i:i+2]) for i in range(0, len(parts) - len(parts) % 2, 2)]
        sentences = [s for s in sentences if s.strip()]
        return self._recombine_broken_sentences(sentences)

    def _recombine_broken_sentences(self, sentences: List[str]) -> List[str]:
        """Fix fragmented sentences (e.g., decimals, markdown)."""
        if len(sentences) < 2:
            return sentences
        open_to_close = {o: c for o, c in self.PAIRED_PUNCT}
        close_to_open = {c: o for o, c in self.PAIRED_PUNCT}
        result = []
        current = ""
        stack = []
        for sent in sentences:
            if current and not (self._check_merge(current, sent) or stack):
                result.append(current)
                current = ""
            for char in sent:
                if char in open_to_close:
                    stack.append(char)
                elif char in close_to_open and stack and stack[-1] == close_to_open[char]:
                    stack.pop()
                current += char
                if char in self.LINE_BREAKS and not stack:
                    result.append(current)
                    current = ""
        if current:
            result.append(current)
        return result

    def _check_merge(self, prev: str, curr: str) -> bool:
        """Determine if two sentence fragments should be merged."""
        if len(prev) < 2 or not curr:
            return False
        last, first = prev[-1], curr[0]
        return (
            (last == "." and prev[-2].isdigit() and first.isdigit()) or
            (last == "." and prev[-2].isdigit() and first not in self.LINE_BREAKS) or
            (last == "!" and prev[-2] in self.LINE_BREAKS and first == "[")
        )

    def _merge_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks respecting token limits."""
        if not sentences:
            return []
        tokens = [len(tokenize_text(s, allowed_special=self._allowed_special)) for s in sentences]
        chunks = []
        start, end, token_count = 0, 0, 0
        while start < len(sentences):
            if end >= len(sentences):
                chunks.append("".join(sentences[start:end]))
                break
            if token_count + tokens[end] <= self._chunk_size:
                token_count += tokens[end]
                end += 1
            else:
                chunks.append("".join(sentences[start:end]))
                overlap = min(self._chunk_overlap, token_count)
                token_count = 0
                while token_count < overlap and end > start:
                    start += 1
                    token_count += tokens[start]
                end = start + 1
                token_count = tokens[start]
        return [c for c in chunks if c.strip()]


def get_summarize_title_keywords(responses: List[Any]) -> List[Tuple[str, str, List[str]]]:
    """Extract title, summary, and keywords from LLM responses."""
    pattern = re.compile(r"\{.*(\}|\]|\,)", re.DOTALL)
    results = []
    for resp in (r.choices[0].message.content for r in responses):
        match = pattern.search(resp)
        if not match:
            results.append(("", "", []))
            continue
        content = match.group(0).strip(",")
        content += "]" * (content.count("[") - content.count("]")) + "}" * (content.count("{") - content.count("}"))
        try:
            data = json.loads(content)
            results.append((data.get("title", ""), data.get("summary", ""), data.get("keywords", [])))
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON: {content}")
            results.append(("", "", []))
    return results