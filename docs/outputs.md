# Outputs

## Item and box pretty printing

Pretty printing of individual consignments and items can be enabled by `--pretty`
in the command line with output like this:

```text
━━ Consignment ━━ Boxes: 3 ━━ Items: 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🐛 ✿ ✿ ✿ ✿ ✿ ✿ 🐛 ✿ ✿ | ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ 🐛 | ✿ ✿ 🐛 ✿ ✿ ✿ ✿ ✿ ✿ 🐛
━━ Consignment ━━ Boxes: 2 ━━ Items: 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🐛 ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ | ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ 🐛
```

Th above is the default output (equivalent with `--pretty=boxes`). Separation of
individual boxes can be disabled using `--pretty=items` where the only unit
shown graphically are the items. Possible output looks like this:

```text
━━ Consignment ━━ Boxes: 3 ━━ Items: 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🐛 ✿ ✿ 🐛 ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ 🐛
━━ Consignment ━━ Boxes: 2 ━━ Items: 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ 🐛
```

Finally, option `--pretty=boxes_only` focuses just on the boxes and does not
show individual items:

```text
── Consignment ── Boxes: 6 ── Items: 60 ───────────────────────────────────
🐛 🐛 ✿ ✿ ✿ 🐛
── Consignment ── Boxes: 4 ── Items: 40 ───────────────────────────────────
✿ ✿ ✿ 🐛
```

This can be further configured using `pretty` key in the configuration file:

```yaml
pretty:
  flower: o
  bug: x
  horizontal_line: "-"
  box_line: "|"
  spaces: false
```

Configuration like the above can allow you to use `--pretty` even when Unicode
characters are properly displayed in your terminal. Note that some characters,
such as the dash (`-`) above, need to be in quotes because they have a special
meaning in YAML.

The output with the settings above will look like:

```text
-- Consignment -- Boxes: 4 -- Items: 40 -----------------------------------
xooooooooo|ooooooxooo|oooooooooo|xoooooooox
-- Consignment -- Boxes: 6 -- Items: 60 -----------------------------------
xxooooooxx|oooooooooo|oooxooooxx|ooooooooxo|ooooxooooo|ooooxoooox
```

## Output details of items and indexes inspected

Alternatively, if you want to plot and further analyze the simulated consignments
and inspections, use `detailed=True` in the `run_scenarios()` function to
return an object with arrays of items (binary values representing contaminated or
not) and the item indexes inspected. If multiple simulation iterations
are being used, the details from the first simulation will be used.

```python
num_consignments = 10
results, details = run_scenarios(
    config=config,
    scenario_table=scenario_table,
    seed=42,
    num_simulations=1,
    num_consignments=num_consignments,
    detailed=True,
)
```

See the `scenarios` documentation and the validation plots Jupyter notebook
(validation_plots.ipynb) for further details on how to use `run_scenarios()` and
the details object. The items and inspected indexes are visualized in the
notebook to confirm that the consignment are being contaminated and inspected as
configured.

If using a command line interface to run the simulation, the flags `--detailed`
or `-d` can be used to print the details object in the terminal. However, using
the details object in the terminal is not recommended as it includes the items
and indexes inspected for the entire simulation and may be very large.


## Replication level output tracking
Outputs specific to the PIS station are managed by the `PISSimData` class
found in `popsborder/outputs.py`. The following are shared constants across
all instances of this class:
  - `PIS_COLUMNS`: This represents the set of columns