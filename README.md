# PoPS Border Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncsu-landscape-dynamics/popsborder/main?urlpath=lab/tree/examples/notebooks/basic_with_command_line.ipynb)
[![CI](https://github.com/ncsu-landscape-dynamics/popsborder/workflows/CI/badge.svg)](https://github.com/ncsu-landscape-dynamics/popsborder/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PoPS Border is a simulation (simulator) of contaminated consignments and
contaminant presence testing which generates synthetic shipment data and
performs inspection on them.

## Model of reality

The simulation is using the following model to understand the system:

```math
f(x) -> y
```

where _x_ represents all shipments with all information about them
such as the level of infestation, _f_ is a sampling function,
i.e. import procedure used at the port,
and _y_ represents the resulting record in the database.

Since the simulation is generating _x_, we can compute:

```math
r = g(y) / g(x)
```

where _x_ and _y_ are defined in the same way as above,
_g_ is a function giving level of infestation in each set
(e.g. number of shipments with a pest),
and _r_ is the success rate in detecting infestation
using the function _f_ from above.

## Use cases

This simulation tool can help to answer various questions about influence
of inspection protocols or pest or contaminant presence on inspection outcome.
For example, the tool can generate synthetic data representing consignments
with variations in contamination rates and test how different inspection
methods influence inspection outcomes.
See more use cases in a dedicated [documentation section](docs/use_cases.md).

The prototype of the simulation was called _pathways-simulation_ because
for some contaminants, such as pests, the main question is what
are the pathways by which the contaminants are getting across the border.

## Examples

An example of how the simulation interface works is in
[this Jupyter notebook](examples/notebooks/basic_with_command_line.ipynb).

To run the code without installing anything use Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncsu-landscape-dynamics/popsborder/main?urlpath=lab/tree/examples/notebooks/basic_with_command_line.ipynb)

If you are not familiar with Binder, see
[our short intro](docs/binder.md).

Documentation is included in the [docs](docs/) directory.
[command line interface](docs/cli.md)
and [Consignment configuration](docs/consignments.md)
pages are good ones to start with.

## Authors

- Vaclav Petras, NCSU Center for Geospatial Analytics
- Kellyn P. Montgomery, NCSU Center for Geospatial Analytics
- Anna Petrasova, NCSU Center for Geospatial Analytics

## License

The simulation code is open source under GNU GPL >=v2
(see the LICENSE file for details).

## Acknowledgment and Disclaimer

This research is funded by USDA APHIS. The findings do not necessarily
represent the views of USDA APHIS.

Please note that this is a simulation and it needs to be calibrated
to give any realistic or actionable results. Results presented here
are examples for demonstration purposes only.


Information on the graphic user interface (GUI) can be found in 
Next: [GUI README](gui/README.md)
