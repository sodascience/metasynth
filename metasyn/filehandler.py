import warnings
from pathlib import Path

import polars as pl
import pyreadstat
import json

_AVAILABLE_FILE_HANDLERS = {}


def filehandler(*args):
    """Register a dataset so that it can be found by name."""

    def _wrap(cls):
        _AVAILABLE_FILE_HANDLERS[cls.name] = cls
        return cls

    return _wrap(*args)

class BaseFileHandler():
    name = "base"
    extensions = []

    def __init__(self, metadata):
        self.metadata = metadata

    def to_dict(self):
        return {
            "file_handler_name": self.name,
            "kwargs": self.metadata,
        }

    def save(self, handler_file):
        with open(handler_file, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=4)

@filehandler
class SavFileHandler(BaseFileHandler):
    name = "spss-sav"
    extensions = [".sav", ".zsav"]

    def read_dataset(self, fp):
        df, _prs_metadata = self._get_df_metadata(fp)
        return df
        # df, prs_metadata = pyreadstat.read_sav(fp, apply_value_formats=True)
        # return pl.DataFrame(df)

    @classmethod
    def _get_df_metadata(cls, fp):
        pandas_df, prs_metadata = pyreadstat.read_sav(fp, apply_value_formats=True)
        df = pl.DataFrame(pandas_df)
        for col in df.columns:
            col_format = prs_metadata.original_variable_types[col]
            if (col_format.startswith("F") and col_format.endswith(".0")
                    and df[col].dtype == pl.Float64):
                df = df.with_columns(pl.col(col).cast(pl.Int64))
        return df, prs_metadata

    @classmethod
    def from_file(cls, fp):
        if Path(fp).suffix not in [".sav", ".zsav"]:
            warnings.warn(f"Trying to read file '{fp}' with extension different from .sav or .zsav")
        if Path(fp).suffix == ".zsav":
            compress = True
        else:
            compress = False
        df, prs_metadata = cls._get_df_metadata(fp)
        # df, prs_metadata = pyreadstat.read_sav(fp, apply_value_formats=True)

        file_label = "This is a synthetic dataset created by metasyn."
        if prs_metadata.file_label is not None:
            file_label += f" Original file label: {file_label}"

        metadata = {
            "column_labels": prs_metadata.column_labels,
            "variable_format": prs_metadata.original_variable_types,
            "compress": compress,
            "variable_display_width": prs_metadata.variable_display_width,
            "file_label": file_label,
            "variable_value_labels": prs_metadata.variable_value_labels,
            "variable_measure": prs_metadata.variable_measure,
        }
        return df, cls(metadata)

    def write_synthetic(self, df, out_fp):
        for col in df.columns:
            col_format = self.metadata["variable_format"][col]
            if (col_format.startswith("F") and not col_format.endswith(".0")
                    and df[col].dtype == pl.Float64):
                n_round = int(col_format.split(".")[-1])
                df = df.with_columns(pl.col(col).round(n_round))
        pyreadstat.write_sav(df.to_pandas(), out_fp, **self.metadata)

@filehandler
class CsvFileHandler(BaseFileHandler):
    name = "csv"
    extensions = [".csv", ".tsv"]

    def read_dataset(self, fp, **kwargs):
        df = pl.read_csv(
            fp, try_parse_dates=True, infer_schema_length=10000,
            null_values=self.null_values,
            ignore_errors=True,
            separator=self.separator,
            quote_char=self.quote_char,
            eol_char=self.eol_char,
            **kwargs)
        return df

    @classmethod
    def from_file(cls, fp, separator=None, eol_char="\n", quote_char='"', null_values=None, **kwargs):
        if Path(fp).suffix == ".tsv" and separator is None:
            separator = "\t"
        if separator is None:
            separator = ","
        if null_values is None:
            null_values = ["", "na", "NA", "N/A", "Na"]
        if isinstance(null_values, str):
            null_values = [null_values]

        df = pl.read_csv(
            fp, try_parse_dates=True, infer_schema_length=10000,
            null_values=null_values,
            ignore_errors=True,
            separator=separator,
            quote_char=quote_char,
            eol_char=eol_char,
            **kwargs)
        metadata = {
            "separator": separator,
            "line_terminator": eol_char,
            "quote_char": quote_char,
            "null_value": null_values[0],
        }
        return df, cls(metadata)

    def write_synthetic(self, df, out_fp):
        df.write_csv(out_fp, **self.metadata)


def get_file_handler(fp):
    suffix = Path(fp).suffix

    for handler_name, handler in _AVAILABLE_FILE_HANDLERS.items():
        if suffix in handler.extensions:
            return handler
    raise ValueError(f"Cannot find handler for files with extension '{suffix}'.")


def load_file_handler(fp):
    with open(fp, "r", encoding="utf-8") as handle:
        metadict = json.load(handle)
    for handler_name, handler_class in _AVAILABLE_FILE_HANDLERS.items():
        if handler_class.name == metadict["file_handler_name"]:
            return handler_class(metadict["kwargs"])
    raise ValueError(f"Cannot find handler with name '{metadict['filehandler']}'")
