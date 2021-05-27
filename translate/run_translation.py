#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
微调seq2seq的库模型。
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

os.environ["WANDB_DISABLED"] = "true"

# 如果没有安装最小版本的transformer，将出现错误。移除的风险由你自己承担。
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)

# 所有需要 src_lang 和 tgt_lang 属性的多语言tokenizer的列表。
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    与我们将要微调的模型/配置/tokenizer相关的参数。
    """

    model_name_or_path: str = field(
        metadata={"help": "来自huggingface.co/models的预训练模型或模型标识符的路径 "}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练模型的配置名称或路径如果None，那么等同于model_name "}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练模型的tokenizer名称或路径如果None，那么等同于model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "本地路径：在哪里存储从HuggingFace.co下载的预训练模模型"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用其中一个fast tokenizer（由tokenizer库支持）"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "要使用的特定模型版本（可以是分支名称、标签名称或提交ID）。"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "将使用运行`transformers-cli login`时生成的token（在私有模型中使用此脚本时必须使用）。"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    与我们要输入模型进行训练和评估的数据有关的参数。
    """
    source_lang: str = field(default=None, metadata={"help": "翻译的源语言ID。 "})
    target_lang: str = field(default=None, metadata={"help": "翻译目标语言ID。 "})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的名称（通过datasets库）"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的配置名称（通过datasets库）。"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "输入训练数据文件（jsonlines）。"})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "一个可选的输入评估数据文件，用于评估jsonlines文件上的指标（sacreblue）"
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "一个可选的输入测试数据文件，用于评估jsonlines文件的指标（sacreblue）。"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集 "}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数量。"},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。"
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。"
        },
    )
    val_max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。将默认为`max_target_length`。"
                    "这个参数也被用来覆盖``model.generate'的``max_length'参数，在``evaluate'和``predict'时使用。 "
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "是否将所有样本Padding到最大句子长度的模型上。如果是False，在批次时将动态地将样本Padding到批中的最大长度。在GPU上更有效，但对TPU非常不利。"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "为了调试的目的或更快的训练，如果设置了训练实例的数量，就将其截断为这个值。"
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "为了调试的目的或更快的训练，如果设置了验证实例的数量，则将其截断为这个值。"
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "为了调试的目的或更快的训练，如果设置了测试实例的数量，就将其截断为这个值。"
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "用于评估的beams的数量。这个参数将被传递给 `model.generate`，在 `evaluate`和 `predict`中使用。"
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "在损失计算中是否忽略与填充标签相对应的token。"
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "在每个源文本前添加一个前缀（对T5模型有用）。"}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "在:obj:`decoder_start_token_id`之后，强制作为第一个生成的token。对于多语言模型，如:doc:`mBART <.../model_doc/mbart>`，第一个生成的token需要是目标语言的token（通常是目标语言token）。"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("需要数据集名称或训练/验证文件。 ")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("需要指定源语言和目标语言。 ")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension == "json", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # 查看SRC/Transformers/Training_args.py中的所有可能参数
    # or by passing the --help flag to this script.
    # 我们现在保留不同的args集，以便更干净地分离关注的参数。

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们只向脚本传递一个参数，而且是一个json文件的路径, 让我们解析它以获得我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "你正在运行一个t5模型，但没有提供一个源前缀，这是必须的的，例如用`--source_prefix 'translate English to German:'"
        )

    # 检测最后一个checkpoint。
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 设置transformer logger的粗略程度为info（仅在主进程中）
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"训练/评估参数 {training_args}")

    # 在初始化模型之前设置random种子。
    set_seed(training_args.seed)

    # 获取数据集：你可以提供你自己的JSON训练和评估文件（见下文）。
    # 或只是提供中心上的一个公共数据集的名称 at https://huggingface.co/datasets/
    # （数据集将自动从数据集hub下载）。
    #
    # 对于翻译，只支持JSON文件，其中有一个名为 "translation"的字段，包含源语言和目标语言的两个key（除非你调整下面的内容）。
    # 在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if data_args.dataset_name == 'custom_zh_en':
        datasets = load_dataset(path='data/custom_zh_en.py', name='custom_zh_en', data_files={'train': ['data/train.cn','data/train.en'], 'validation': ['data/dev.cn','data/dev.en'], 'test': ['data/test.cn','data/test.en']})
    elif data_args.dataset_name is not None:
        # 从hub下载并加载数据集。
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # 查看更多关于加载任何类型的标准或自定义数据集（从文件、python dict、pandas DataFrame等）的信息，请访问
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 2. 加载预训练的模型和tokenizer
    #
    # 分布式训练：
    # .from_pretrained方法保证只有一个本地进程可以同时下载模型和单词表。
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        gradient_checkpointing = True,
        use_cache = False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # 设置解码的开始的token id decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("确保正确定义了`config.decoder_start_token_id` ")
    # T5模型使用
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # 预处理数据集。
    # 我们需要tokenize输入和目标。 column_names: ['translation']
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("没有什么可以做的。 请传入`do_train`，`do_eval`和/或`do_predict`。 ")
        return

    # mBart: 对于翻译，我们设置源语言和目标语言的代码（只对mBART有用，其他的会忽略这些属性
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )
        # tokenizer.src_lang: 'en_XX' ;  tokenizer.tgt_lang: 'ro_RO'
        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # 对于像 mBART-50 和 M2M100 这样的多语言翻译模型，我们需要强制目标语言token作为第一个生成的token。我们要求用户明确提供--forced_bos_token参数。
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # 获取输入/目标的语言code， source_lang：en,  target_lang: ro
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]

    # 临时设置Max_target_Length进行训练。 max_target_length: 128
    max_target_length = data_args.max_target_length
    #是否padding到最长: False
    padding = "max_length" if data_args.pad_to_max_length else False
    #标签平滑因子
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        """
        # 对数据进行预处理
        Args:
            examples ():  {'translation': [{'en':xxx, 'ro':yyyy},....]}  一个translation里面包含1千条训练样本对
            一条训练样本对 eg: {'en': 'Membership of Parliament: see Minutes', 'ro': 'Componenţa Parlamentului: a se vedea procesul-verbal'}
        Returns:

        """
        #取出所有的src语句和trg语句，形成列表
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        # 如果是T5模型，需要加前缀
        inputs = [prefix + inp for inp in inputs]
        #max_source_length： 输入序列的的最大长度 , 返回tokenid和attention mask。  model_inputs.data : {'input_ids':[[xx,xx],...], 'attention_mask':[[1,1],...}
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # 为目标设置token程序， labels也是和model_inputs同样的格式
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # 如果我们在这里进行填充，当我们想忽略损失中的填充时，将标签中所有tokenizer.pad_token_id替换为-100。
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            # 是否从训练样本中截取部分，一般测试用
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # 数据的 collator 函数： label_pad_token_id: -100
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # 评估指标 Metric
    if os.path.exists('data/sacrebleu.py'):
        metric = load_metric('data/sacrebleu.py')
    else:
        metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # 初始化我们的Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
