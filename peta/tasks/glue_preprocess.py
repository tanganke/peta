from .datasets_preprocess import DatasetPreprocessor, preprocess


class CoLA_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/cola
    """

    def preprocess(self, sentence: str, label: int):
        assert isinstance(sentence, str)
        assert isinstance(label, int)
        input_text = "cola sentence: {}".format(sentence)
        if label in [0, 1]:
            target_text = "acceptable" if label == 1 else "unacceptable"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["sentence"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence, label in zip(example["sentence"], example["label"]):
                _input_text, _target_text = self.preprocess(sentence, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class RTE_Preprocess(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/rte
    """

    def __call__(self, example):
        """
        Preprocess the RTE dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1]
        input_text = "rte sentence1: {sentence1} sentence2: {sentence2}".format(
            **example
        )
        target_text = "entailment" if example["label"] == 0 else "not_entailment"

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class MNLI_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/mnli/
    """

    def __call__(self, example):
        """
        Preprocess the MNLI dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1, 2], "invalid label, must be 0, 1, or 2"
        input_text = "mnli hypothesis: {hypothesis} premise: {premise}".format(
            hypothesis=example["hypothesis"], premise=example["premise"]
        )
        target_text = {0: "entailment", 1: "neutral", 2: "contradiction"}[
            example["label"]
        ]

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class MRPC_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/mrpc
    """

    def __call__(self, example):
        """
        Preprocess the MRPC dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1]
        input_text = "mrpc sentence1: {sentence1} sentence2: {sentence2}".format(
            **example
        )
        target_text = "not_equivalent" if example["label"] == 0 else "equivalent"

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class QNLI_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/qnli
    """

    def __call__(self, example):
        """
        Preprocess the QNLI dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1]
        input_text = "qnli question: {question} sentence: {sentence}".format(**example)
        target_text = "not_entailment" if example["label"] == 1 else "entailment"

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class QQP_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/qqp
    """

    def __call__(self, example):
        """
        Preprocess the QQP dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1]
        input_text = "qqp question1: {question1} question2: {question2}".format(
            **example
        )
        target_text = "not_duplicate" if example["label"] == 0 else "duplicate"

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class SST2_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/sst2
    """

    def __call__(self, example):
        """
        Preprocess the SST2 dataset into a text-to-text format.
        """
        assert example["label"] in [0, 1]
        input_text = "sst2 sentence: {}".format(example["sentence"])
        target_text = "negative" if example["label"] == 0 else "positive"

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class STSB_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/stsb
    """

    def __call__(self, example):
        """
        Preprocess the STSB dataset into a text-to-text format.
        """
        input_text = "stsb sentence1: {sentence1} sentence2: {sentence2}".format(
            **example
        )
        target_text = "{:.1f}".format(example["label"])

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )
