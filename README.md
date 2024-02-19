# SEAL: Self-Explanation Assessment Library 

---

This codebase corresponds to several papers which will be listed below once published.
* ["Automated Assessment of Comprehension Strategies from Self-Explanations Using LLMs"](https://www.mdpi.com/2078-2489/14/10/567)

The repository contains several branches that were used for development. All stable code is gradually being moved to main. **We strongly advise using only the main branch code.**

## Install dependencies

The project requires Python3.8. In order to install the dependencies, the command below should be enough:
```pip install -r requirements.txt```

Access to the training corpus can be provided upon request.
We recommend creating a `data` folder in the root of the project, and placing the file containing the corpus there. <br>
Pretrained LoRA adaptors for running the `flan_test.py` script can be found here: [link](https://huggingface.co/nbogdan)

If you wish to use Singularity for setting up the running environment, the `ses38.def` file can be used.

## Training and evaluation

The scripts present in the `scripts` folder can be used for replicating functionality described in the papers.
All scripts rely on downloading the dataset previously and placing it in the data folder.
The scripts will train and validate with reduced verbosity. At the end of a validation run statistics will be logged. 2 types of statistics will be logged:
    * Optimistic: In which badly formatted LLM answer will be assumed as class 0. (This type of validation was used in the paper)
    * Pessimistic: In which badly formatted LLM answer will not be considered in the statistics.

* **flan_k_shot.py**
  * Can be used for 'out-of-the-box' FLAN-T5 runs, using pretrained models from Huggingface.
* **flan_train.py**
  * Can be used for training FLAN-T5 runs, using pretrained models from Huggingface. The following variables can be 'played' with for different usecases:
    * upload_adapter: True means that the LoRA adapter will be uploaded to Huggingface, **if the Huggingface was done with a valid api key (marked "secret" in the code)**
    * batch_size: Can be increased to train with a larger batch_size and speed up the run. It was set by default to 1, to reduce the risk of an out-of-memory crash if larger models were used.
    * epochs: Represents the number of epochs for the fine-tuning process. The default is 1.
* **flan_test.py**
  * This script can be used for running experiments by loading trained LoRA adapters
  * The adapters that we make available have the following format:
    * {model_name}-{model_size}-{num_examples}ex-{task}-{num_epochs}epochs
* **gpt_eval.py**
  * Can be used to run evaluations using gpt3.5-turbo-boost
  * **Note: An OpenAI account and API_KEY are required for running. The API key must be placed in this variable: `openai.api_key`**