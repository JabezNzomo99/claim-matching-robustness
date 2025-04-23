from rolaser_embed import SentenceEncoder
from text_processing import PreprocessLine
from custom_tokenizers import CustomTokenizer


class RoLaserEncoder(SentenceEncoder):
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        vocab=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
        tokenizer="spm",
    ):
        super().__init__(
            model_path=model_path,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            spm_vocab=vocab,
            cpu=cpu,
            fp16=fp16,
            verbose=verbose,
            sort_kind=sort_kind,
        )
        self.tokenizer = CustomTokenizer(tokenizer)

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        preprocessed_sentences = [PreprocessLine(s) for s in sentences]
        tokenized_sentences = [
            self.tokenizer.tokenize(s) for s in preprocessed_sentences
        ]
        embeddings = super().encode_sentences(
            tokenized_sentences, normalize_embeddings=True
        )
        return embeddings
