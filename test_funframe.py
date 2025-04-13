import unittest
import funframe as testFrame
from pathlib import Path

data_dir = Path.cwd() / "datasets"


sample_datasets = [
    data_dir / "iris.csv",
    data_dir / "iris_no_header.csv",
    data_dir / "iris_alternative_sep.csv",  # uses ? instead of comma
    data_dir / "all_types.csv",
]


class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.df = testFrame.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]}
        )

    def test_columns(self):
        self.assertEqual(self.df.columns(), {"a", "b", "c"})

    def test_has_column(self):
        self.assertTrue(self.df.has_column("b"))
        self.assertFalse(self.df.has_column("d"))

    def test_add_column(self):
        new_df = self.df.add_column("d", [7, 8, 9])
        self.assertIn("d", new_df.columns())
        self.assertEqual(list(new_df.to_dict()["d"]), [7, 8, 9])

    def test_remove_column(self):
        new_df = self.df.remove_column("b")
        self.assertNotIn("b", new_df.columns())

    def test_map_column(self):
        new_df = self.df.map_column("a", "doubled", lambda x: x * 2)
        self.assertIn("doubled", new_df.columns())
        self.assertEqual(list(new_df.to_dict()["doubled"]), [2, 4, 6])

    def test_select_rows(self):
        selected = self.df.select_rows([0, 2])
        self.assertEqual(list(selected.to_dict()["a"]), [1, 3])

    def test_filter_on_column(self):
        filtered = self.df.filter_on_column("a", lambda x: x > 1)
        self.assertEqual(list(filtered.to_dict()["a"]), [2, 3])

    def test_combine_columns(self):
        combined = self.df.combine_columns("sum_ab", "a", "b", lambda x, y: x + y)
        self.assertEqual(list(combined.to_dict()["sum_ab"]), [5, 7, 9])

    def test_fold_column(self):
        result = self.df.fold_column("a", lambda acc, x: acc + x, 0)
        self.assertEqual(result, 6)

    def test_sum_column(self):
        self.assertEqual(self.df.sum_column("a"), 6.0)

    def test_prod_column(self):
        self.assertEqual(self.df.prod_column("a"), 6.0)

    def test_mean_column(self):
        self.assertEqual(self.df.mean_column("a"), 2.0)

    def test_shape(self):
        self.assertEqual(self.df.shape(), (3, 3))

    def test_to_dict(self):
        d = self.df.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(list(d["a"]), [1, 2, 3])

    def test_column_storage_types(self):
        d = self.df.to_dict()
        self.assertEqual(type(d["a"]).__name__, "array")
        self.assertEqual(type(d["b"]).__name__, "array")
        self.assertEqual(type(d["c"]).__name__, "list")


class TestDataFrameFromCSV(unittest.TestCase):
    def test_iris_csv(self):
        df = testFrame.DataFrame.from_csv(sample_datasets[0])
        self.assertEqual(df.shape()[0], 5)  # Expecting 5 columns
        self.assertGreater(df.shape()[1], 0)  # Expecting at least 1 row

    def test_iris_no_header(self):
        df = testFrame.DataFrame.from_csv(sample_datasets[1], has_header=False)
        self.assertTrue(all(col.startswith("col_") for col in df.columns()))
        self.assertGreater(df.shape()[1], 0)

    def test_alternative_sep(self):
        df = testFrame.DataFrame.from_csv(sample_datasets[2], delimiter="?")
        self.assertGreater(len(df.columns()), 1)
        self.assertGreater(df.shape()[1], 0)

    def test_all_types_csv(self):
        df = testFrame.DataFrame.from_csv(sample_datasets[3])
        self.assertIn("name", df.columns())
        self.assertIn("age", df.columns())
        self.assertIn("weight", df.columns())
        self.assertIn("is_active", df.columns())
        self.assertEqual(df.shape()[0], 4)


if __name__ == "__main__":
    unittest.main()
