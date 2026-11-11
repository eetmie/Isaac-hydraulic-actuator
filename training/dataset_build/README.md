# Dataset build record

The exact scripts and sources that built `dataset/complete_dataset`
([Hugging Face](https://huggingface.co/datasets/eetmie/excavator-hydraulic-actuator)),
kept byte-for-byte: the C sources match `source_code_sha256` in its `dataset.json`.
They are a record, not a maintained tool. Input paths are hard-coded to the original
machines (Desktop CSV exports and the robot's `actuator-data` folder), so they do not run
elsewhere as is.

`PROCESSING.md` (the dataset's original README) and `FUSION_AUDIT.md` describe the processing and
the fusion audit; the JSON files are the source inventories, firmware hash comparison and audit
results. They are unchanged, so their `provenance/` paths mean this folder.

- `build_complete_dataset.py`, `expand_gen2_dataset.py`, `gen2_fusion.py`: September export,
  then the October robot recordings and gen2 orientation replay.
- `validate_gen2_dataset.py`: checks that produced `validation.json`.
- `september_export/`: the September Desktop inventory, build, validation and tests, with the
  `pico_v2` replay sources used for it.
- `audited_gen2_fusion/`: the audited firmware fusion replay, including cold-start and firmware
  references. Its `Fusion/` library is identical to [`training/fusion/Fusion`](../fusion/Fusion)
  and is not repeated; the maintained replay is `training.offline_fusion`.

The vendored x-io Fusion library is MIT licensed (`september_export/pico_v2/Fusion/LICENSE.md`).
Compiled replay binaries are not kept; they build from these sources.
