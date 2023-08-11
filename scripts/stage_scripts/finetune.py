import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import json

from dvc.api import params_show

from src.models.finetune_mlm.finetune import finetune

if __name__ == "__main__":
    params = params_show()
    finetuning_params = params["training"]["finetuning"]

    with open(params["paths"]["train_sents"]) as f:
        train_sents = json.load(f)

    finetune(
        initial_name_or_path=params["paths"]["expanded_vocab_model"],
        save_path=params["paths"]["finetuned_model"],
        train_sents=train_sents,
        batch_size=finetuning_params["batch_size"],
        lr=finetuning_params["lr"],
        mlm_probability=finetuning_params["mlm_probability"],
        num_epochs=finetuning_params["num_epochs"],
        shedular_name=finetuning_params["shedular_name"],
        verbose=True,
    )
