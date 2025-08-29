import re


class StringHelper:
    """Helper class for string operations."""

    @staticmethod
    def get_no_sentences(content: str) -> int:
        """Counts the number of sentences in the given content."""

        # Split on ., !, or ? followed by whitespace or end of string.
        sentences = re.split(r"[.!?]+\s*", content)

        return len([s for s in sentences if s.strip()])
