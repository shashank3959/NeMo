Multi-LoRA inference with NVIDIA NIM
====================================

This is a demonstration of deploying multiple LoRA adapters with NVIDIA
NIM. NIM supports LoRA adapters in .nemo (from NeMo Framework), and
Hugging Face model formats.

We will deploy the PubMedQA LoRA adapter from previous notebook,
alongside two other previously trained LoRA adapters (GSM8K, SQuAD) that
are available on NVIDIA NGC as examples.

``NOTE``: While it’s not necessary to complete the LoRA training and
obtain the adapter from the previous notebook (“Creating a LoRA adapter
with NeMo Framework”) to follow along with this one, it is recommended
if possible. You can still learn about LoRA deployment with NIM using
the other adapters downloaded from NGC.

This notebook includes instructions to send an inference call to NVIDIA
NIM using the Python ``requests`` library.

Before you begin
----------------

Ensure that you satisfy the pre-requisites, and have completed the setup
instructions provided in the README associated with this tutorial.

--------------

.. code:: ipython3

    import requests
    import json

Check available LoRA models
---------------------------

Once the NIM server is up and running, check the available models as
follows:

.. code:: ipython3

    url = 'http://0.0.0.0:8000/v1/models'
    
    response = requests.get(url)
    data = response.json()
    
    print(json.dumps(data, indent=4))

This will return all the models available for inference by NIM. In this
case, it will return the base model ``meta/llama3-8b-instruct``, as well
as the LoRA adapters that were provided during NIM deployment -
``llama3-8b-pubmed-qa`` (if applicable),
``llama3-8b-instruct-lora_vnemo-math-v1``, and
``llama3-8b-instruct-lora_vnemo-squad-v1``. Note that their names match
the folder names where their .nemo files are stored.

--------------

Multi-LoRA inference
--------------------

Inference can be performed by sending POST requests to the
``/completions`` endpoint.

A few things to note: \* The ``model`` parameter in the payload
specifies the model that the request will be directed to. This can be
the base model ``meta/llama3-8b-instruct``, or any of the LoRA models,
such as ``llama3-8b-pubmed-qa``. \* ``max_tokens`` parameter specifies
the maximum number of tokens to generate. At any point, the cumulative
number of input prompt tokens and specified number of output tokens to
generate should not exceed the model’s maximum context limit. For
llama3-8b-instruct, the context length supported is 8192 tokens.

Following code snippets show how it’s possible to send requests
belonging to different LoRAs (or tasks). NIM dynamically loads the LoRA
adapters and serves the requests. It also internally handles the
batching of requests belonging to different LoRAs to allow better
performance and more efficient of compute.

PubMedQA
~~~~~~~~

If you have trained the PubMedQA LoRA model and made it available via
NIM inference, try sending an example from the test set.

.. code:: ipython3

    url = 'http://0.0.0.0:8000/v1/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Example from the PubMedQA test set
    prompt="BACKGROUND: Sublingual varices have earlier been related to ageing, smoking and cardiovascular disease. The aim of this study was to investigate whether sublingual varices are related to presence of hypertension.\nMETHODS: In an observational clinical study among 431 dental patients tongue status and blood pressure were documented. Digital photographs of the lateral borders of the tongue for grading of sublingual varices were taken, and blood pressure was measured. Those patients without previous diagnosis of hypertension and with a noted blood pressure \u2265 140 mmHg and/or \u2265 90 mmHg at the dental clinic performed complementary home blood pressure during one week. Those with an average home blood pressure \u2265 135 mmHg and/or \u2265 85 mmHg were referred to the primary health care centre, where three office blood pressure measurements were taken with one week intervals. Two independent blinded observers studied the photographs of the tongues. Each photograph was graded as none/few (grade 0) or medium/severe (grade 1) presence of sublingual varices. Pearson's Chi-square test, Student's t-test, and multiple regression analysis were applied. Power calculation stipulated a study population of 323 patients.\nRESULTS: An association between sublingual varices and hypertension was found (OR = 2.25, p<0.002). Mean systolic blood pressure was 123 and 132 mmHg in patients with grade 0 and grade 1 sublingual varices, respectively (p<0.0001, CI 95 %). Mean diastolic blood pressure was 80 and 83 mmHg in patients with grade 0 and grade 1 sublingual varices, respectively (p<0.005, CI 95 %). Sublingual varices indicate hypertension with a positive predictive value of 0.5 and a negative predictive value of 0.80.\nQUESTION: Is there a connection between sublingual varices and hypertension?\n ### ANSWER (yes|no|maybe): "
    
    data = {
        "model": "llama3-8b-pubmed-qa",
        "prompt": prompt,
        "max_tokens": 128
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    
    print(json.dumps(response_data, indent=4))

.. code:: ipython3

    response

Grade School Math (GSM8K dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    url = 'http://0.0.0.0:8000/v1/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    prompt = '''Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer:'''
    
    data = {
        "model": "llama3-8b-instruct-lora_vnemo-math-v1",
        "prompt": prompt,
        "max_tokens": 128
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    
    print(json.dumps(response_data, indent=4))

Extractive Question-Answering (SQuAD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    url = 'http://0.0.0.0:8000/v1/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    prompt = '''CONTEXT: "The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. The Normans were famed for their martial spirit and eventually for their Christian piety, becoming exponents of the Catholic orthodoxy into which they assimilated. They adopted the Gallo-Romance language of the Frankish land they settled, their dialect becoming known as Norman, Normaund or Norman French, an important literary language. The Duchy of Normandy, which they formed by treaty with the French crown, was a great fief of medieval France, and under Richard I of Normandy was forged into a cohesive and formidable principality in feudal tenure. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy on the Saracens and Byzantines, and an expedition on behalf of their duke, William the Conqueror, led to the Norman conquest of England at the Battle of Hastings in 1066. Norman cultural and military influence spread from these new European centres to the Crusader states of the Near East, where their prince Bohemond I founded the Principality of Antioch in the Levant, to Scotland and Wales in Great Britain, to Ireland, and to the coasts of north Africa and the Canary Islands.\nQUESTION: What were the Norman dynasty famous for? ANSWER:'''
    data = {
        "model": "llama3-8b-instruct-lora_vnemo-squad-v1",
        "prompt": prompt,
        "max_tokens": 128
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    
    print(json.dumps(response_data, indent=4))

--------------

(Optional) Testing the accuracy of NIM inference
------------------------------------------------

If you followed the previous notebook on training a Llama-3-8b-Instruct
LoRA adapter using NeMo Framework and evaluated the model accuracy, you
can test the same using NIM inference for validation.

.. code:: ipython3

    # Ensure that the path to PubMedQA test data is correct
    data_test = json.load(open("./pubmedqa/data/test_set.json",'rt'))
    
    def read_jsonl (fname):
        obj = []
        with open(fname, 'rt') as f:
            st = f.readline()
            while st:
                obj.append(json.loads(st))
                st = f.readline()
        return obj
    
    prepared_test = read_jsonl("./pubmedqa/data/pubmedqa_test.jsonl")

.. code:: ipython3

    # Send an inference request to the PubMedQA LoRA model
    def infer(prompt):
    
        url = 'http://0.0.0.0:8000/v1/completions'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
        data = {
            "model": "llama3-8b-pubmed-qa",
            "prompt": prompt,
            "max_tokens": 128
        }
    
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
    
        return(response_data["choices"][0]["text"])

.. code:: ipython3

    from tqdm import tqdm
    
    results = {}
    sample_id = list(data_test.keys())
    
    for i, key in tqdm(enumerate(sample_id)):
        answer = infer(prepared_test[i]['input'].strip())
        answer = answer.lower()
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

    answer

.. code:: ipython3

    # dump results
    FILENAME="pubmedqa-llama-3-8b-lora-NIM.json"
    with(open(FILENAME, "w")) as f:
        json.dump(results, f)
    
    # Evaluation
    !cp $FILENAME ./pubmedqa/
    !cd ./pubmedqa/ && python evaluation.py $FILENAME

NIM inference should provide comparable accuracy to NeMo Framework
inference.

Note that each individual answer also conform to the format we
specified, i.e. ``<<< {answer} >>>``.
