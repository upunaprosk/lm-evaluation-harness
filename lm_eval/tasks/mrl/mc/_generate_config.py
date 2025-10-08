from pathlib import Path

import datasets


if __name__ == "__main__":
    subsets = [
        x
        for x in datasets.get_dataset_config_names(
            "mrlbenchmarks/global-piqa-nonparallel"
        )
        if not x.startswith("dev")
    ]
    PARENT = Path(__file__).parent
    for s in subsets:
        with open(PARENT / f"{s}.yaml", "w") as f:
            f.write("include: '_template_mc'\n")
            f.write(f"task: mrl_{s}\n")
            f.write(f"dataset_name: {s}\n")
