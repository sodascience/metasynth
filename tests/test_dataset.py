from pathlib import Path

import pandas as pd
from metasynth.dataset import MetaDataset
from metasynth.var import MetaVar
import pytest


def test_dataset():
    dtypes = {
        "PassengerId": "int",
        "Survived": "category",
        "Pclass": "category",
        "Name": "string",
        "Sex": "category",
        "SibSp": "category",
        "Parch": "category",
        "Ticket": "string",
        "Cabin": "string",
        "Embarked": "category",
        "Age": "float",
        "Fare": "float",
    }
    titanic_fp = Path("tests", "data", "titanic.csv")
    tmp_fp = Path("tests", "data", "tmp.json")
    df = pd.read_csv(titanic_fp, dtype=dtypes)
    dataset = MetaDataset.from_dataframe(df)

    def check_dataset(dataset):
        assert dataset.n_columns == 12
        assert dataset.n_rows == 891
        sub_types = {
            "int": "discrete",
            "string": "string",
            "category": "categorical",
            "float": "continuous",
        }
        for col, type_name in dtypes.items():
            meta_var_type = sub_types[type_name]
            assert dataset[col].var_type == meta_var_type
        with pytest.raises(KeyError):
            dataset["unknown"]
        with pytest.raises(IndexError):
            dataset[20]
        with pytest.raises(TypeError):
            dataset[MetaVar]

        assert "Rows" in str(dataset)

    check_dataset(dataset)
    dataset.to_json(tmp_fp)
    dataset = MetaDataset.from_json(tmp_fp)
    check_dataset(dataset)
