Creating a Llama-3 LoRA adapter with NeMo Framework
===================================================

This notebook showcases performing LoRA PEFT **Llama 3 8B** on
`PubMedQA <https://pubmedqa.github.io/>`__ using NeMo Framework.
PubMedQA is a Question-Answering dataset for biomedical texts.

   ``NOTE:`` Ensure that you run this notebook inside the `NeMo
   Framework
   container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`__
   which has all the required dependencies. Instructions are available
   in the associated tutorial README.

.. code:: ipython3

    !pip install ipywidgets

--------------

Step-by-step instructions
-------------------------

This notebook is structured into six steps: 1. Download
Llama-3-8B-Instruct from Hugging Face 2. Convert Llama-3-8B-Instruct to
NeMo format 3. Prepare the dataset 4. Run the PEFT finetuning script 5.
Inference with NeMo Framework 6. Check the model accuracy

Step 1: Download the model from Hugging Face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ``NOTE:`` Access to Meta-Llama-3-8B-Instruct is gated. Before you
   proceed, ensure that you have a Hugging Face account, and have
   requested the necessary permission from Hugging Face and Meta to
   download the model on the
   `Meta-Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__
   page. Then, you can use your Hugging Face `access
   token <https://huggingface.co/docs/hub/en/security-tokens>`__ to
   download the model in the following code snippet, which we will then
   convert and customize with NeMo Framework.

.. code:: ipython3

    import os
    import huggingface_hub
    
    # Set your Hugging Face access token
    huggingface_hub.login("<YOUR_HUGGINGFACE_ACCESS_TOKEN>")

.. code:: ipython3

    os.makedirs("./Meta-Llama-3-8B-Instruct" ,exist_ok=True)
    huggingface_hub.snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", local_dir="Meta-Llama-3-8B-Instruct", local_dir_use_symlinks=False)

The Llama-3-8B-Instruct model will be downloaded to
``./Meta-Llama-3-8B-Instruct``

Step 2: Convert Llama-3-8B-Instruct to NeMo format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the below code to convert the model to the NeMo format.

The generated ``.nemo`` file uses distributed checkpointing and can be
loaded with any Tensor Parallel (TP) or Pipeline Parallel (PP)
combination without reshaping or splitting. For more information on
parallelisms in NeMo, refer to `NeMo Framework
documentation <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html>`__.

.. code:: bash

    %%bash
    
    # clear any previous temporary weights dir if any
    rm -r model_weights
    
    python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
      --precision bf16 \
      --input_name_or_path=./Meta-Llama-3-8B-Instruct/ \
      --output_path=./Meta-Llama-3-8B-Instruct.nemo

This will create a .nemo model file in current working directory.

Step 3: Prepare the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the PubMedQA dataset and run the pre-processing script in the
cloned directory.

.. code:: bash

    %%bash
    
    # Download the dataset and prep. scripts
    git clone https://github.com/pubmedqa/pubmedqa.git
    
    # split it into train/val/test datasets
    cd pubmedqa/preprocess
    python split_dataset.py pqal

The following example shows what a single row looks inside of the
PubMedQA train, validation and test splits.

.. code:: json

   "18251357": {
       "QUESTION": "Does histologic chorioamnionitis correspond to clinical chorioamnionitis?",
       "CONTEXTS": [
           "To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother.",
           "A retrospective review was performed on 52 cases with a histologic diagnosis of acute chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as controls. Cases and controls were selected sequentially. Maternal medical records were reviewed for indicators of maternal infection.",
           "Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = 0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more clinical indicators was significantly associated with the presence of histologic chorioamnionitis (p = 0.019)."
       ],
       "reasoning_required_pred": "yes",
       "reasoning_free_pred": "yes",
       "final_decision": "yes",
       "LONG_ANSWER": "Histologic chorioamnionitis is a reliable indicator of infection whether or not it is clinically apparent."
   },

Use the following code to convert the train, validation, and test
PubMedQA data into the ``JSONL`` format that NeMo needs for PEFT.

.. code:: ipython3

    import json
    
    def read_jsonl(fname):
        obj = []
        with open(fname, 'rt') as f:
            st = f.readline()
            while st:
                obj.append(json.loads(st))
                st = f.readline()
        return obj
    
    def write_jsonl(fname, json_objs):
        with open(fname, 'wt') as f:
            for o in json_objs:
                f.write(json.dumps(o)+"\n")
                
    def form_question(obj):
        st = ""    
        for i, label in enumerate(obj['LABELS']):
            st += f"{label}: {obj['CONTEXTS'][i]}\n"
        st += f"QUESTION: {obj['QUESTION']}\n"
        st += f" ### ANSWER (yes|no|maybe): "
        return st
    
    def convert_to_jsonl(data_path, output_path):
        data = json.load(open(data_path, 'rt'))
        json_objs = []
        for k in data.keys():
            obj = data[k]
            prompt = form_question(obj)
            completion = obj['final_decision']
            json_objs.append({"input": prompt, "output": f"<<< {completion} >>>"})
        write_jsonl(output_path, json_objs)
        return json_objs
    
    
    test_json_objs = convert_to_jsonl("pubmedqa/data/test_set.json", "pubmedqa/data/pubmedqa_test.jsonl")
    train_json_objs = convert_to_jsonl("pubmedqa/data/pqal_fold0/train_set.json", "pubmedqa/data/pubmedqa_train.jsonl")
    dev_json_objs = convert_to_jsonl("pubmedqa/data/pqal_fold0/dev_set.json", "pubmedqa/data/pubmedqa_val.jsonl")

   ``Note:`` In the output, we enforce the inclusion of “<<<” and “>>>“
   markers which would allow verification of the LoRA tuned model during
   inference. This is because the base model can produce “yes” / “no”
   responses based on zero-shot templates as well.

.. code:: ipython3

    # clear up cached mem-map file
    !rm pubmedqa/data/*idx*

After running the above script, you will see ``pubmedqa_train.jsonl``,
``pubmedqa_val.jsonl``, and ``pubmedqa_test.jsonl`` files appear in the
data directory.

This is what an example will be formatted like after the script has
converted the PubMedQA data into ``JSONL`` -

.. code:: json

   {"input": "QUESTION: Failed IUD insertions in community practice: an under-recognized problem?\nCONTEXT: The data analysis was conducted to describe the rate of unsuccessful copper T380A intrauterine device (IUD) insertions among women using the IUD for emergency contraception (EC) at community family planning clinics in Utah.\n ...  ### ANSWER (yes|no|maybe): ",
   "output": "<<< yes >>>"}

Step 4: Run PEFT finetuning script for LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo framework includes a high level python script for fine-tuning
`megatron_gpt_finetuning.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py>`__
that can abstract away some of the lower level API calls. Once you have
your model downloaded and the dataset ready, LoRA fine-tuning with NeMo
is essentially just running this script!

For this demonstration, this training run is capped by ``max_steps``,
and validation is carried out every ``val_check_interval`` steps. If the
validation loss does not improve after a few checks, training is halted
to avoid overfitting.

   ``NOTE:`` In the block of code below, pass the paths to your train,
   test and validation data files as well as path to the .nemo model.

.. code:: bash

    %%bash
    
    # Set paths to the model, train, validation and test sets.
    MODEL="./Meta-Llama-3-8B-Instruct.nemo"
    TRAIN_DS="[./pubmedqa/data/pubmedqa_train.jsonl]"
    VALID_DS="[./pubmedqa/data/pubmedqa_val.jsonl]"
    TEST_DS="[./pubmedqa/data/pubmedqa_test.jsonl]"
    TEST_NAMES="[pubmedqa]"
    
    SCHEME="lora"
    TP_SIZE=1
    PP_SIZE=1
    
    OUTPUT_DIR="./results/Meta-Llama-3-8B-Instruct"
    rm -r $OUTPUT_DIR
    
    torchrun --nproc_per_node=1 \
    /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        exp_manager.explicit_log_dir=${OUTPUT_DIR} \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        trainer.precision=bf16-mixed \
        trainer.val_check_interval=20 \
        trainer.max_steps=500 \
        model.megatron_amp_O2=True \
        ++model.mcore_gpt=True \
        model.tensor_model_parallel_size=${TP_SIZE} \
        model.pipeline_model_parallel_size=${PP_SIZE} \
        model.micro_batch_size=1 \
        model.global_batch_size=8 \
        model.restore_from_path=${MODEL} \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.validation_ds.file_names=${VALID_DS} \
        model.peft.peft_scheme=${SCHEME}

This will create a LoRA adapter - a file named
``megatron_gpt_peft_lora_tuning.nemo`` in
``./results/Meta-Llama-3-8B-Instruct/checkpoints/``. We’ll use this
later.

To further configure the run above -

-  **A different PEFT technique**: The ``peft.peft_scheme`` parameter
   determines the technique being used. In this case, we did LoRA, but
   NeMo Framework supports other techniques as well - such as P-tuning,
   Adapters, and IA3. For more information, refer to the `PEFT support
   matrix <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/peft/landing_page.html>`__.
   For example, for P-tuning, simply set

.. code:: bash

   model.peft.peft_scheme="ptuning" # instead of "lora"

-  **Tuning Llama-3 70B**: You will need 8xA100 or 8xH100 GPUs. Provide
   the path to it’s .nemo checkpoint (similar to the download and
   conversion steps earlier), and change the model parallelization
   settings for Llama-3 70B PEFT to distribute across the GPUs. It is
   also recommended to run the fine-tuning script from a terminal
   directly instead of Jupyter when using more than 1 GPU.

.. code:: bash

   model.tensor_model_parallel_size=8
   model.pipeline_model_parallel_size=1

You can override many such configurations while running the script. A
full set of possible configurations is located in `NeMo Framework
Github <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml>`__.

Step 5: Inference with NeMo Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running text generation within the framework is also possible with
running a Python script. Note that is more for testing and validation,
not a full-fledged deployment solution like NVIDIA NIM.

.. code:: ipython3

    # Check that the LORA model file exists
    !ls -l ./results/Meta-Llama-3-8B-Instruct/checkpoints

In the code snippet below, the following configurations are worth noting
-

1. ``model.restore_from_path`` to the path for the
   Meta-Llama-3-8B-Instruct.nemo file.
2. ``model.peft.restore_from_path`` to the path for the PEFT checkpoint
   that was created in the fine-tuning run in the last step.
3. ``model.test_ds.file_names`` to the path of the pubmedqa_test.jsonl
   file

If you have made any changes in model or experiment paths, please ensure
they are configured correctly below.

.. code:: bash

    %%bash
    MODEL="./Meta-Llama-3-8B-Instruct.nemo"
    TEST_DS="[./pubmedqa/data/pubmedqa_test.jsonl]"
    TEST_NAMES="[pubmedqa]"
    SCHEME="lora"
    TP_SIZE=1
    PP_SIZE=1
    
    # This is where your LoRA checkpoint was saved
    PATH_TO_TRAINED_MODEL="./results/Meta-Llama-3-8B-Instruct/checkpoints/megatron_gpt_peft_lora_tuning.nemo"
    
    # The generation run will save the generated outputs over the test dataset in a file prefixed like so
    OUTPUT_PREFIX="pubmedQA_result_"
    
    python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
        model.restore_from_path=${MODEL} \
        model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        model.data.test_ds.file_names=${TEST_DS} \
        model.data.test_ds.names=${TEST_NAMES} \
        model.data.test_ds.global_batch_size=1 \
        model.data.test_ds.micro_batch_size=1 \
        model.data.test_ds.tokens_to_generate=3 \
        model.tensor_model_parallel_size=${TP_SIZE} \
        model.pipeline_model_parallel_size=${PP_SIZE} \
        inference.greedy=True \
        model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
        model.data.test_ds.write_predictions_to_file=True

Step 6: Check the model accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the results are in, let’s read the results and calculate the
accuracy on the pubmedQA task. You can compare your accuracy results
with the public leaderboard at https://pubmedqa.github.io/.

Let’s take a look at one of the predictions in the generated output
file. The ``pred`` key indicates what was generated.

.. code:: ipython3

    !tail -n 1 pubmedQA_result__test_pubmedqa_inputs_preds_labels.jsonl

Note that the model produces output in the specified format, such as
``<<< no >>>``.

The following snippet loads the generated output and calculates accuracy
in comparison to the test set using the ``evaluation.py`` script
included in the PubMedQA repo.

.. code:: ipython3

    import json
    
    answers = []
    with open("pubmedQA_result__test_pubmedqa_inputs_preds_labels.jsonl",'rt') as f:
        st = f.readline()
        while st:
            answers.append(json.loads(st))
            st = f.readline()

.. code:: ipython3

    data_test = json.load(open("./pubmedqa/data/test_set.json",'rt'))

.. code:: ipython3

    results = {}
    sample_id = list(data_test.keys())
    
    for i, key in enumerate(sample_id):
        answer = answers[i]['pred']
        if 'yes' in answer:
            results[key] = 'yes'
        elif 'no' in answer:
            results[key] = 'no'
        elif 'maybe' in answer:
            results[key] = 'maybe'
        else:
            print("Malformed answer: ", answer)
            results[key] = 'maybe'

.. code:: ipython3

    # Dump results in a format that can be ingested by PubMedQA evaluation file
    FILENAME="pubmedqa-llama-3-8b-lora.json"
    with(open(FILENAME, "w")) as f:
        json.dump(results, f)
    
    # Evaluation
    !cp $FILENAME ./pubmedqa/
    !cd ./pubmedqa/ && python evaluation.py $FILENAME

For the Llama-3-8B-Instruct model, you should see accuracy comparable to
the below:

::

   Accuracy 0.786000
   Macro-F1 0.550305
