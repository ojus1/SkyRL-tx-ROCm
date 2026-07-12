"""One-step variant of Tinker Cookbook's ``sl_basic`` recipe for local SkyRL."""

import asyncio
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name="qwen3",
        max_length=128,
        batch_size=1,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common,
        file_path=str(WORKSPACE / "tinker-cookbook/tinker_cookbook/example_data/conversations.jsonl"),
    )
    config = train.Config(
        log_path=str(WORKSPACE / "runs/sl_basic_qwen3_0.6b"),
        model_name=MODEL,
        recipe_name="recipe_sl_basic_local_smoke",
        renderer_name="qwen3",
        dataset_builder=dataset,
        learning_rate=2e-4,
        lr_schedule="constant",
        num_epochs=1,
        lora_rank=8,
        base_url="http://127.0.0.1:8000",
        save_every=0,
        eval_every=0,
        infrequent_eval_every=0,
        max_steps=1,
        submit_ahead=0,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
