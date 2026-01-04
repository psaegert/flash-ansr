import unittest
import torch

from flash_ansr import Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = Tokenizer(vocab=["a", "b", "c", "d"])

        assert len(tokenizer) == 11

        assert tokenizer.encode(["a", "b", "c", "d"]) == [7, 8, 9, 10]
        assert tokenizer.decode([7, 8, 9, 10]) == ["a", "b", "c", "d"]
        assert tokenizer.decode(torch.tensor([7, 8, 9, 10])) == ["a", "b", "c", "d"]

        assert tokenizer.encode(["a", "b", "c", "d"], return_tensors=True).tolist() == [7, 8, 9, 10]

        assert tokenizer["a"] == 7
        assert tokenizer[7] == "a"

        assert "a" in tokenizer
        assert 7 in tokenizer

        assert list(tokenizer) == ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>", "<constant>", "a", "b", "c", "d"]

    def test_tokenizer_from_config(self):
        config = {
            "tokenizer": {
                "variables": ["a", "b", "c", "d"],
                "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<my_special_token>"],
                "operators": ["+", "-", "*", "/"],
            }
        }

        tokenizer = Tokenizer.from_config(config)

        assert len(tokenizer) == 14

        assert tokenizer.encode(["a", "b", "c", "d"]) == [10, 11, 12, 13]
        assert tokenizer.decode([10, 11, 12, 13]) == ["a", "b", "c", "d"]

        assert tokenizer.encode(["a", "b", "c", "d"], return_tensors=True).tolist() == [10, 11, 12, 13]

        assert tokenizer["a"] == 10
        assert tokenizer[10] == "a"

        assert "a" in tokenizer
        assert 10 in tokenizer

        assert list(tokenizer) == ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<my_special_token>", "+", "-", "*", "/", "a", "b", "c", "d"]

    def test_tokenizer_invalid(self):
        tokenizer = Tokenizer(vocab=["a", "b", "c", "d"])

        with self.assertRaises(KeyError):
            tokenizer["non_existent_token"]

        with self.assertRaises(KeyError):
            tokenizer[123456789]

        with self.assertRaises(TypeError):
            tokenizer[1.0]

        with self.assertRaises(TypeError):
            1.0 in tokenizer
