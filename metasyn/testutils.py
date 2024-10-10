"""Module for testing the functionality of distributions and providers.

The testutils module provides a set of utilities for testing the functionality
and internal consistency of individual distributions and providers.
"""


from __future__ import annotations

import json

import jsonschema
import numpy as np
import polars as pl
from jsonschema.exceptions import SchemaError

from metasyn.distribution import MultinoulliDistribution, NADistribution
from metasyn.distribution.base import BaseDistribution
from metasyn.metaframe import _jsonify
from metasyn.privacy import BasePrivacy
from metasyn.provider import (
    BaseDistributionProvider,
    DistributionProviderList,
    get_distribution_provider,
)
from metasyn.var import MetaVar


def check_distribution_provider(provider_name: str):
    """Check internal consistency of a distribution provider.

    Arguments
    ---------
    provider_name:
        Name of the provider to be tested.
    """
    provider = get_distribution_provider(provider_name)
    assert isinstance(provider, BaseDistributionProvider)
    assert len(provider.distributions) > 0
    assert all(issubclass(dist, BaseDistribution) for dist in provider.distributions)
    assert isinstance(provider.name, str)
    assert len(provider.name) > 0
    assert provider.name == provider_name
    assert isinstance(provider.version, str)
    assert len(provider.version) > 0


def check_distribution(distribution: type[BaseDistribution], privacy: BasePrivacy,
                       provenance: str):
    """Check whether the distributions in the package can be validated positively.

    Arguments
    ---------
    distribution:
        Distribution to validate to check whether it behaves as expected.
    privacy:
        Level/type of privacy the distribution adheres to.
    provenance:
        Which provider/plugin/package provides the distribution.
    """
    # Check the schema of the distribution.
    schema = distribution.schema()
    dist_dict = distribution.default_distribution().to_dict()
    try:
        jsonschema.validate(_jsonify(dist_dict), schema)
    except SchemaError as err:
        raise ValueError(f"Failed distribution validation for {distribution.__name__}") from err

    # Check the privacy
    assert privacy.is_compatible(distribution)
    if isinstance(distribution.var_type, str):
        var_types = [distribution.var_type]
    else:
        var_types = distribution.var_type
    for vt in var_types:
        DistributionProviderList(provenance).find_distribution(
            distribution.implements, var_type=vt, privacy=privacy,
            unique=distribution.unique)

    assert len(distribution.implements.split(".")) == 2
    assert distribution.provenance == provenance
    assert distribution.var_type != "unknown"
    dist = distribution.default_distribution()
    series = pl.Series([dist.draw() for _ in range(100)])
    new_dist = distribution.fit(series, **privacy.fit_kwargs)
    assert isinstance(new_dist, distribution)
    assert set(list(new_dist.to_dict())) >= set(
        ("implements", "provenance", "class_name", "parameters"))



HEADER_TEMPLATE = """
# Metasyn report for {file_name}

## General

- Number of rows: {n_rows}
- Number of columns: {n_columns}
- Generated by {program_name} version {program_version} at {generation_time}

## Variables / columns

"""


VAR_TEMPLATE = """
### {var_name}

- Distribution {class_name} with parameters:
{parameters}
- Variable type {var_type}
- Based on {n_based_on} values{disclosure}
- Percentage missing: {missing_perc} %
- Generated examples: {example_list}


"""

def create_md_report(file_name, out_md_file):
    """Create markdown report from GMF file."""
    with open(file_name, "r") as handle:
        gmf_dict = json.load(handle)

    header = HEADER_TEMPLATE.format(
        file_name=file_name,
        n_rows=gmf_dict["n_rows"],
        n_columns=gmf_dict["n_columns"],
        program_name=gmf_dict["provenance"]["created by"]["name"],
        program_version=gmf_dict["provenance"]["created by"]["version"],
        generation_time=gmf_dict["provenance"]["creation time"],
    )
    variables = ""

    for var_dict in gmf_dict["vars"]:
        var = MetaVar.from_dict(var_dict)
        if isinstance(var.distribution, MultinoulliDistribution):
            parameters = [f"\t- {label}: {round(prob*gmf_dict['n_rows']*(1-var.prop_missing))}\n"
                          for label, prob in zip(var.distribution.labels, var.distribution.probs)]
        elif isinstance(var.distribution, NADistribution):
            variables += (f"### {var.name}\n- Distribution NADistribution\n- Only missing values"
                          "\n- Examples: NA, NA, NA, ...\n")
            continue
        else:
            parameters = [f"\t - {name}: {value}\n"
                             for name, value in var.distribution._param_dict().items()]
        parameter_str = "".join(parameters)
        if var.prop_missing > 0:
            examples = np.random.permutation([str(var.distribution.draw()) for _ in range(3)] +
                                             ["NA", "NA"])
        else:
            examples = [str(x) for x in var.draw_series(5)]

        if "privacy" in var_dict["creation_method"]:
            partition_size = var_dict["creation_method"]["privacy"]["parameters"]["partition_size"]
            disclosure = f" using micro aggregation with a partition size of {partition_size}"
        else:
            disclosure = ""
        variables += VAR_TEMPLATE.format(
            var_name = var.name,
            var_type=var.var_type,
            class_name=var.distribution.__class__.__name__,
            n_based_on=round(gmf_dict["n_rows"]*(1-var.prop_missing)),
            example_list=", ".join(examples) + ", ...",
            parameters=parameter_str,
            missing_perc=f"{100*var.prop_missing:.2f}",
            disclosure=disclosure,
        )
    with open(out_md_file, "w", encoding="utf-8") as handle:
        handle.write(header + variables)


def create_input_toml(file_name):
    """Create input toml with all distribution in builtin."""
    import tomlkit

    prov = get_distribution_provider("builtin")
    doc = tomlkit.document()
    doc.add("config_version", "1.1")
    doc.add("dist_providers", ["builtin"])
    doc.add("n_rows", 100)
    doc.add("defaults", {"data_free": True, "prop_missing": 0.1})
    var_array = tomlkit.aot()
    for dist in prov.distributions:
        var = tomlkit.table()
        var.add("name", dist.__name__)
        if isinstance(dist.var_type, str):
            var_type = dist.var_type
        else:
            var_type = dist.var_type[0]
        var.add("var_type", var_type)
        dist_dict = _jsonify(dist.default_distribution().to_dict())
        dist_dict.pop("version")
        dist_dict.pop("class_name")
        dist_dict.pop("provenance")
        var.add("distribution", dist_dict)
        var.add(tomlkit.nl())
        var_array.append(var)
    doc.add("var", var_array)

    with open(file_name, "w", encoding="utf-8") as handle:
        tomlkit.dump(doc, handle)
