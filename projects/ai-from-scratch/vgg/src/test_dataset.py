import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dataset import classification_loaders


class ClassificationDataTest(unittest.TestCase):
    def test_split_and_partial_batch_without_auxiliary_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            lines = []
            for breed in range(2):
                for index in range(5):
                    name = f"breed{breed}_{index}"
                    Image.new("RGB", (40, 35), (100, 120, 140)).save(root / "images" / f"{name}.jpg")
                    lines.append(f"{name} {breed + 1} 1 {breed + 1}\n")
            (root / "annotations/trainval.txt").write_text("".join(lines))
            train, val, names = classification_loaders(root, image_size=32, batch_size=3)
            train_ids, val_ids = set(train.dataset.indices), set(val.dataset.indices)
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(train_ids | val_ids, set(range(10)))
            self.assertEqual(len(names), 2)
            self.assertEqual([len(labels) for _, labels in train], [3, 3, 2])
            images, labels = next(iter(val))
            self.assertEqual(tuple(images.shape), (2, 32, 32, 3))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertTrue(torch.isfinite(images).all())
            _, repeated_val, _ = classification_loaders(root, image_size=32, batch_size=3)
            self.assertEqual(val.dataset.indices, repeated_val.dataset.indices)


if __name__ == "__main__":
    unittest.main()
