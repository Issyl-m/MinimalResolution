# MinimalResolution [UNDER CONSTRUCTION]

Proof-of-concept of computing minimal resolutions over the mod 2 Steenrod algebra written by me. This program constructs a free resolution of $\mathbb{F}_2$ as an $\mathcal{A}_2-$module.

# Usage

1. Execute `minr.py`.
2. Run `chart.py` to obtain a plot resembling the computed `Ext groups`.

Note that we are not only interested in these `Ext groups`. This computation has much more information saved in `list_admissible_generators_by_deg`, `list_found_generators `, `list_new_generator_mapping_table`, and `list_differentials` (i.e. the differentials in the resolution). This information could be used to compute further multiplicative structures.

# Parameters

The default length of the minimal resolution is hardcoded in `MAX_NUMBER_OF_ROWS` (see `config.py`). 
