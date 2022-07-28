from abc import abstractmethod
from typing import List, Union
import warnings
import inspect
import pkg_resources

import numpy as np

from metasynth.distribution.base import BaseDistribution
from metasynth.distribution.discrete import DiscreteUniformDistribution,\
    PoissonDistribution, UniqueKeyDistribution
from metasynth.distribution.continuous import UniformDistribution,\
    NormalDistribution, LogNormalDistribution, TruncatedNormalDistribution
from metasynth.distribution.categorical import CatFreqDistribution
from metasynth.distribution.regex.base import RegexDistribution,\
    UniqueRegexDistribution
from metasynth.distribution.faker import FakerDistribution
from metasynth.distribution.datetime import UniformDateDistribution,\
    UniformTimeDistribution


class BaseDistributionTree():
    @property
    @abstractmethod
    def discrete_distributions(self) -> List[type]:
        """Get the integer distributions."""

    @property
    @abstractmethod
    def continuous_distributions(self) -> List[type]:
        """Get continuous distributions."""

    @property
    @abstractmethod
    def categorical_distributions(self) -> List[type]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def string_distributions(self) -> List[type]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def date_distributions(self) -> List[type]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def time_distributions(self) -> List[type]:
        """Get categorical distributions."""

    @property
    @abstractmethod
    def datetime_distributions(self) -> List[type]:
        """Get categorical distributions."""

    def get_dist_list(self, var_type: str) -> List[type]:
        prop_str = var_type + "_distributions"
        if not hasattr(self, prop_str):
            raise ValueError(f"Unknown variable type '{var_type}' detected.")
        return getattr(self, prop_str)

    def fit(self, series, var_type, unique=False):
        dist_list = self.get_dist_list(var_type)
        if len(dist_list) == 0:
            raise ValueError(f"No available distributions with variable type: '{var_type}'")
        dist_instances = [d.fit(series) for d in dist_list]
        dist_aic = [d.information_criterion(series) for d in dist_instances]
        i_best_dist = np.argmin(dist_aic)
        warnings.simplefilter("always")
        if dist_instances[i_best_dist].is_unique and unique is None:
            warnings.warn(f"\nVariable {series.name} seems unique, but not set to be unique.\n"
                          "Set the variable to be either unique or not unique to remove this "
                          "warning.\n")
        if unique is None:
            return dist_instances[i_best_dist]

        dist_aic = [dist_aic[i] for i in range(len(dist_aic))
                    if dist_instances[i].is_unique == unique]
        dist_instances = [d for d in dist_instances if d.is_unique == unique]
        if len(dist_instances) == 0:
            raise ValueError(f"No available distributions for variable '{series.name}'"
                             f" with variable type '{var_type}' "
                             f"that have unique == {unique}.")
        return dist_instances[np.argmin(dist_aic)]

    @property
    def all_var_types(self):
        return [p[:-14] for p in dir(self.__class__)
                if isinstance(getattr(self.__class__, p), property) and p.endswith("_distributions")
                ]

    def find_distribution(self, dist_name):
        for var_type in self.all_var_types:
            for dist_class in self.get_dist_list(var_type):
                if dist_class.is_named(dist_name):
                    return dist_class, {}
        raise ValueError(f"Cannot find distribution with name '{dist_name}'.")

    def fit_distribution(self, dist, series):
        dist_instance = None
        if isinstance(dist, str):
            dist_class, fit_kwargs = self.find_distribution(dist)
            dist_instance = dist_class.fit(series, **fit_kwargs)
        elif inspect.isclass(dist) and issubclass(dist, BaseDistribution):
            dist_instance = dist.fit(series)
        if isinstance(dist, BaseDistribution):
            dist_instance = dist

        if dist_instance is None:
            raise TypeError(
                f"Distribution with type {type(dist)} is not a BaseDistribution")

        return dist_instance

    def from_dict(self, var_dict):
        for dist_class in self.get_dist_list(var_dict["type"]):
            if dist_class.is_named(var_dict["distribution"]["name"]):
                return dist_class.from_dict(var_dict["distribution"])
        raise ValueError(f"Cannot find distribution with name '{var_dict['distribution']['name']}'"
                         f"and type '{var_dict['type']}'.")


class BuiltinDistributionTree(BaseDistributionTree):
    @property
    def discrete_distributions(self) -> List[type]:
        return [DiscreteUniformDistribution, PoissonDistribution, UniqueKeyDistribution]

    @property
    def continuous_distributions(self) -> List[type]:
        return [UniformDistribution, NormalDistribution, LogNormalDistribution,
                TruncatedNormalDistribution]

    @property
    def categorical_distributions(self) -> List[type]:
        return [CatFreqDistribution]

    @property
    def string_distributions(self) -> List[type]:
        return [RegexDistribution, UniqueRegexDistribution, FakerDistribution]

    @property
    def date_distributions(self) -> List[type]:
        return [UniformDateDistribution]

    @property
    def time_distributions(self) -> List[type]:
        return [UniformTimeDistribution]

    @property
    def datetime_distributions(self) -> List[type]:
        return [UniformDateDistribution]


def get_disttree(target: Union[str, type, BaseDistributionTree]=None) -> BaseDistributionTree:
    if target is None:
        target = "builtin"
    if isinstance(target, BaseDistributionTree):
        return target
    if isinstance(target, type):
        return target()

    all_disttrees = {
        entry.name: entry
        for entry in pkg_resources.iter_entry_points("metasynth.disttree")
    }
    try:
        return all_disttrees[target].load()()
    except KeyError as exc:
        raise ValueError(f"Cannot find distribution tree with name '{target}'.") from exc
