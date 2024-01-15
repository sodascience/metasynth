"""Module defining the MetaVar class, which represents a metadata variable."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from metasyn.distribution.base import BaseDistribution
from metasyn.privacy import BasePrivacy, BasicPrivacy
from metasyn.provider import BaseDistributionProvider, DistributionProviderList


class MetaVar():
    """Metadata variable.

    MetaVar is a structure that holds all metadata needed to generate a
    synthetic column for it. This is the variable level building block for the
    MetaFrame. It contains the methods to convert a polars Series into a
    variable with an appropriate distribution. The MetaVar class is to the
    MetaFrame what a polars Series is to a DataFrame.

    This class is considered a passthrough class used by the MetaFrame class,
    and is not intended to be used directly by the user.

    Parameters
    ----------
    var_type:
        String containing the variable type, e.g. continuous, string, etc.
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
    description:
        User-provided description of the variable.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 name: str,
                 var_type: str,
                 distribution: BaseDistribution,
                 dtype: str = "unknown",
                 description: Optional[str] = None,
                 prop_missing: float = 1.0):
        self.name = name
        self.var_type = var_type
        self.distribution = distribution
        self.dtype = dtype
        self.description = description
        self.prop_missing = prop_missing
            # series = _to_polars(series)
            # self.name = series.name
            # if prop_missing is None:
            #     self.prop_missing = (
            #         len(series) - len(series.drop_nulls())) / len(series)
            # self.dtype = str(series.dtype)

        # self.series = series
        # self.description = description

        # if self.prop_missing is None:
        #     raise ValueError(f"Error while initializing variable {self.name}."
        #                      " prop_missing is None.")
        if self.prop_missing < -1e-8 or self.prop_missing > 1+1e-8:
            raise ValueError(f"Cannot create variable '{self.name}' with proportion missing "
                             "outside range [0, 1]")

    # @classmethod
    # def detect(cls,
    #            series_or_dataframe: Union[pd.Series,
    #                                       pl.Series,
    #                                       pl.DataFrame],
    #            description: Optional[str] = None,
    #            prop_missing: Optional[float] = None):
    #     """Detect variable class(es) of series or dataframe.

<<<<<<< HEAD
        This method does not fit any distribution, but it does infer the
        correct types for the MetaVar and saves the Series for later fitting.

        Parameters
        ----------
        series_or_dataframe: pd.Series or pd.Dataframe
            If the variable is a pandas Series, then find the correct
            variable type and create an instance of that variable.
            If a Dataframe is supplied instead, a list of of variables is
            returned: one for each column in the dataframe.
        description:
            User description of the variable.
        prop_missing:
            Proportion of the values missing. If None, detect it from the series.
            Otherwise prop_missing should be a float between 0 and 1.
=======
    #     Parameters
    #     ----------
    #     series_or_dataframe: pd.Series or pd.Dataframe
    #         If the variable is a pandas Series, then find the correct
    #         variable type and create an instance of that variable.
    #         If a Dataframe is supplied instead, a list of of variables is
    #         returned: one for each column in the dataframe.
    #     description:
    #         User description of the variable.
    #     prop_missing:
    #         Proportion of the values missing. If None, detect it from the series.
    #         Otherwise prop_missing should be a float between 0 and 1.
>>>>>>> 2ce6998 (Update according to discussion)

    #     Returns
    #     -------
    #     MetaVar:
    #         It returns a meta data variable of the correct type.
    #     """
    #     if isinstance(series_or_dataframe, (pl.DataFrame, pd.DataFrame)):
    #         if isinstance(series_or_dataframe, pd.DataFrame):
    #             return [MetaVar.detect(series_or_dataframe[col])
    #                     for col in series_or_dataframe]
    #         return [MetaVar.detect(series) for series in series_or_dataframe]

    #     series = _to_polars(series_or_dataframe)
    #     var_type = cls.get_var_type(series)

    #     return cls(var_type, series, description=description, prop_missing=prop_missing)

    @staticmethod
    def get_var_type(series: pl.Series) -> str:
        """Convert polars dtype to metasyn variable type.

        This method uses internal polars methods, so this might break at some
        point.

        Parameters
        ----------
        series:
            Series to get the metasyn variable type for.

        Returns
        -------
        var_type:
            The variable type that is found.
        """
        try:
            polars_dtype = pl.datatypes.dtype_to_py_type(series.dtype).__name__
        except NotImplementedError:
            polars_dtype = pl.datatypes.dtype_to_ffiname(series.dtype)

        convert_dict = {
            "int": "discrete",
            "float": "continuous",
            "date": "date",
            "datetime": "datetime",
            "time": "time",
            "str": "string",
            "categorical": "categorical"
        }
        try:
            return convert_dict[polars_dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported polars type '{polars_dtype}") from exc

    def to_dict(self) -> Dict[str, Any]:
        """Create a dictionary from the variable."""
        if self.distribution is None:
            dist_dict = {}
        else:
            dist_dict = self.distribution.to_dict()
        var_dict = {
            "name": self.name,
            "type": self.var_type,
            "dtype": self.dtype,
            "prop_missing": self.prop_missing,
            "distribution": dist_dict,
        }
        if self.description is not None:
            var_dict["description"] = self.description
        return var_dict

    def __str__(self) -> str:
        """Return an easy to read formatted string for the variable."""
        description = f'Description: "{self.description}"\n' if self.description else ""

        if self.distribution is None:
            distribution_formatted = "No distribution information available"
        else:
            distribution_formatted = "\n".join(
                "\t" + line for line in str(self.distribution).split("\n")
            )

        return (
            f'"{self.name}"\n'
            f'{description}'
            f'- Variable Type: {self.var_type}\n'
            f'- Data Type: {self.dtype}\n'
            f'- Proportion of Missing Values: {self.prop_missing:.4f}\n'
            f'- Distribution:\n{distribution_formatted}\n'
        )

    @classmethod
    def fit(cls,  # pylint: disable=too-many-arguments
            series: pl.Series,
            dist_spec: Optional[dict] = None,
            provider_list: DistributionProviderList = DistributionProviderList("builtin"),
            privacy: BasePrivacy = BasicPrivacy(),
            prop_missing: Optional[float] = None,
            description: Optional[str] = None):
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
        dist_providers:
            Distribution providers that are used for fitting.
        privacy:
            Privacy level to use for fitting the series.
        unique:
            Whether the variable should be unique. If not supplied, it will be
            inferred from the data.
        fit_kwargs:
            Extra options for distributions during the fitting stage.
        """
        var_type = cls.get_var_type(series)
        if isinstance(dist_spec, BaseDistribution):
            dist_spec = dist_spec.to_dict()
        elif isinstance(dist_spec, type):
            dist_spec = {"implements": dist_spec.implements, "unique": dist_spec.is_unique}
        elif isinstance(dist_spec, str):
            dist_spec = {"implements": dist_spec}
        elif dist_spec is None:
            dist_spec = {}

        distribution = provider_list.fit(series, var_type, dist_spec, privacy)
        if prop_missing is None:
            prop_missing = (len(series) - len(series.drop_nulls())) / len(series)
        return cls(series.name, var_type, distribution=distribution, dtype=str(series.dtype),
                   description=description, prop_missing=prop_missing)

    def draw(self) -> Any:
        """Draw a random item for the variable in whatever type is required."""
        # if self.distribution is None:
            # raise ValueError("Cannot draw without distribution")

        # Return NA's -> None
        if self.prop_missing is not None and np.random.rand() < self.prop_missing:
            return None
        return self.distribution.draw()

    def draw_series(self, n: int) -> pl.Series:
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
        # if not isinstance(self.distribution, BaseDistribution):
            # raise ValueError("Cannot draw without distribution.")
        self.distribution.draw_reset()
        value_list = [self.draw() for _ in range(n)]
        if "Categorical" in self.dtype:
            return pl.Series(value_list, dtype=pl.Categorical)
        return pl.Series(value_list)

    @classmethod
    def from_dict(cls,
                  var_dict: Dict[str, Any],
                  distribution_providers: Union[
                      None, str, type[BaseDistributionProvider],
                      BaseDistributionProvider] = None) -> MetaVar:
        """Restore variable from dictionary.

        Parameters
        ----------
        distribution_providers:
            Distribution providers to use to create the variable. If None,
            use all installed/available distribution providers.
        var_dict:
            This dictionary contains all the variable and distribution
            information to recreate it from scratch.

        Returns
        -------
        MetaVar:
            Initialized metadata variable.
        """
        provider_list = DistributionProviderList(distribution_providers)
        dist = provider_list.from_dict(var_dict)
        return cls(
            name=var_dict["name"],
            var_type=var_dict["type"],
            distribution=dist,
            prop_missing=var_dict["prop_missing"],
            dtype=var_dict["dtype"],
            description=var_dict.get("description", None)
        )


def _to_polars(series: Union[pd.Series, pl.Series]) -> pl.Series:
    if isinstance(series, pl.Series):
        return series
    if len(series.dropna()) == 0:
        series = pl.Series(name=series.name,
                           values=[None for _ in range(len(series))])
    else:
        series = pl.Series(series)
    return series
