"""Variable module that creates metadata variables."""  # pylint: disable=invalid-name

from __future__ import annotations

import inspect
from typing import Union, Dict, List, Any

import pandas as pd
import numpy as np

from metasynth.metadist import MetaDistribution
from metasynth.distribution.base import BaseDistribution
from metasynth.distribution.util import get_dist_class
from metasynth.distribution.util import _get_all_distributions


class MetaVar():
    """Meta data variable.

    Acts as a base class for specific types of variables, but also as a
    launching pad for detecting its type.

    Parameters
    ----------
    var_type:
        Variable type as a string, e.g. continuous, string, etc.
    series:
        Series to create the variable from. Series is None by default and in
        this case the value is ignored. If it is not supplied, then the
        variable cannot be fit.
    name:
        Name of the variable/column.
    distribution:
        Distribution to draw random values from. Can also be set by using the
        fit method.
    prop_missing:
        Proportion of the series that are missing/NA.
    dtype:
        Type of the original values, e.g. int64, float, etc. Used for type-casting
        back.
    """

    dtype = "unknown"

    def __init__(self,  # pylint: disable=too-many-arguments
                 var_type: str,
                 series: pd.Series=None,
                 name: str=None,
                 distribution: BaseDistribution=None,
                 prop_missing: float=0,
                 dtype: str=None):
        self.var_type = var_type
        if series is None:
            self.name = name
            self.prop_missing = prop_missing
            if dtype is not None:
                self.dtype = dtype
        else:
            self.name = series.name
            self.prop_missing = (len(series) - len(series.dropna()))/len(series)
            self.dtype = str(series.dtype)

        self.series = series
        self.distribution = distribution

    @classmethod
    def detect(cls, series_or_dataframe):
        """Detect variable class(es) of series or dataframe.

        Parameters
        ----------
        series_or_dataframe: pd.Series or pd.Dataframe
            If the variable is a pandas Series, then find the correct
            variable type and create an instance of that variable.
            If a Dataframe is supplied instead, a list of of variables is
            returned: one for each column in the dataframe.

        Returns
        -------
        MetaVar:
            It returns a meta data variable of the correct type.
        """
        if isinstance(series_or_dataframe, pd.DataFrame):
            return [MetaVar.detect(series_or_dataframe[col])
                    for col in series_or_dataframe]

        series = series_or_dataframe
        return cls(cls.get_var_type(pd.api.types.infer_dtype(series)), series)

    @staticmethod
    def get_var_type(pandas_dtype: str) -> str:
        """Convert pandas dtype to MetaSynth variable type."""
        convert_dict = {
            "categorical": "categorical",
            "string": "string",
            "integer": "discrete",
            "floating": "continuous",
            "date": "date",
            "datetime64": "datetime",
            "time": "time",
        }
        try:
            return convert_dict[pandas_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported pandas type '{pandas_dtype}") from exc

    def to_dict(self) -> Dict[str, Any]:
        """Create a dictionary from the variable."""
        if self.distribution is None:
            dist_dict = {}
        else:
            dist_dict = self.distribution.to_dict()
        return {
            "name": self.name,
            "type": self.var_type,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": dist_dict,
        }

    def __str__(self) -> str:
        """Create a readable string from a variable."""
        return str({
            "name": self.name,
            "type": type(self).__name__,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": str(self.distribution),
        })

    def fit(self, dist: Union[str, BaseDistribution, type]=None,
            distribution_tree: Dict[str, List[BaseDistribution]]=None,
            unique=None):
        """Fit distributions to the data.

        If multiple distributions are available for the current data type,
        use the one that fits the data the best.

        While it has no arguments or return values, it will set the
        distribution attribute to the most suitable distribution.

        Parameters
        ----------
        dist:
            The distribution to fit. In case of a string, search for it
            using the aliases of all distributions. Otherwise use the
            supplied distribution (class). Examples of allowed strings are:
            "normal", "uniform", "faker.city.nl_NL". If not supplied, fit
            the best available distribution for the variable type.
        distribution_tree:
            Distribution tree to be used.
            By default use all distributions in metasynth.distribution.
        unique:
            Whether the variable should be unique. If not supplied, it will be
            inferred from the data.
        """
        if self.series is None:
            raise ValueError("Cannot fit distribution if we don't have the"
                             "original data.")

        # Automatic detection of the distribution
        if distribution_tree is None:
            distribution_tree = _get_all_distributions("metasynth.distribution")

        # Manually supplied distribution
        fit_kwargs: Dict[str, Any] = {}
        if dist is None:
            dist_instance = MetaDistribution.fit(self.series, distribution_tree[self.var_type],
                                                 unique=unique)
        if isinstance(dist, str):
            dist_class, fit_kwargs = get_dist_class(dist)
            dist_instance = dist_class.fit(self.series, **fit_kwargs)
        elif inspect.isclass(dist) and issubclass(dist, BaseDistribution):
            dist_instance = dist.fit(self.series)
        if isinstance(dist, BaseDistribution):
            dist_instance = dist
        # if not isinstance(dist_instance, BaseDistribution):
        try:
            self.distribution = dist_instance
        except UnboundLocalError as exc:
            raise TypeError(
                f"Distribution with type {type(dist)} is not a BaseDistribution") from exc

    def draw(self) -> Any:
        """Draw a random item for the variable in whatever type is required."""
        if self.distribution is None:
            raise ValueError("Cannot draw without distribution")

        # Return NA's -> None
        if np.random.rand() < self.prop_missing:
            return None
        return self.distribution.draw()

    def draw_series(self, n: int) -> pd.Series:
        """Draw a new synthetic series from the metadata.

        Parameters
        ----------
        n:
            Length of the series to be created.

        Returns
        -------
        pandas.Series:
            Pandas series with the synthetic data.
        """
        if not isinstance(self.distribution, BaseDistribution):
            raise ValueError("Cannot draw without distribution.")
        self.distribution.draw_reset()
        return pd.Series([self.draw() for _ in range(n)], dtype=self.dtype)

    @classmethod
    def from_dict(cls, var_dict: Dict[str, Any]) -> MetaVar:
        """Restore variable from dictionary.

        Parameters
        ----------
        var_dict:
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        all_distributions = _get_all_distributions("metasynth.distribution")
        if var_dict["type"] not in all_distributions:
            raise ValueError(f"Cannot find variable type '{var_dict['type']}")
        distributions = all_distributions[var_dict["type"]]
        for dist_class in distributions:
            if dist_class.is_named(var_dict["distribution"]["name"]):
                return cls(
                    var_dict["type"],
                    name=var_dict["name"],
                    distribution=dist_class.from_dict(var_dict["distribution"]),
                    prop_missing=var_dict["prop_missing"], dtype=var_dict["dtype"])
        raise ValueError(f"Cannot find meta class '{var_dict['type']}'.")
