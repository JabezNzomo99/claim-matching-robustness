# When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits 
[![Static Badge](https://img.shields.io/badge/Paper-arXiv%3A2503.03417-brightgreen?logoColor=Blue)
](https://arxiv.org/abs/2503.03417) <a href="#bibtex"><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

[Jabez Magomere](), [Emanuele La Malfa](https://emanuelelm.github.io/), [Manuel Tonneau](https://manueltonneau.com/), [Ashkan Kazemi](https://ashkankzme.github.io/), [Scott Hale](https://www.oii.ox.ac.uk/people/profiles/scott-hale/)

This repository contains the code for the paper [*When Claims Evolve*: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits](https://arxiv.org/abs/2503.03417). If you have any questions, feel free to create a Github issue or reach out to the first author at jabez.magomere@keble.ox.ac.uk. 

## Overall Description
A visual summary of our approach is provided below:

<p align="center">
  <img src="assets/our_approach.png" alt="Our Approach" style="width: 100%;">
</p>

<!-- ### Before Reranking Results on *CheckThat22* Dataset
<p align="center">
  <img src="assets/before_reranking_results_all_CheckThat2022_plot.png" alt="Our Approach" style="width: 100%;">
</p> -->

### After Reranking Results on *CheckThat22* Dataset

<p align="center">
  <img src="assets/after_reranking_results_all_CheckThat2022_plot.png" alt="Our Approach" style="width: 100%;">
</p>

## 📦 Getting Started
### 1. 🐍 Set up the Conda environment

```bash
conda env create -f environment.yml
conda activate your-env-name  # replace with the name in environment.yml
```

### 2. 🛠️ Install the project

```bash
python -m pip install .
```

### 3. ⚙️ Set Up Environment Variables

Create a `.env.local` file in the project root and add your OpenAI API key for LLM-based generations:

```dotenv
OPENAI_API_KEY=your_openai_key_here
```

### 4. 🚀 Define Script Aliases (Optional)

For quicker script execution, you can load predefined aliases:

```bash
source aliases.sh
```
## 📂 Datasets Access

This project uses three datasets, each stored locally in a standardized structure inspired by the [TREC format](https://trec.nist.gov/). All datasets are formatted with the following components:

- A `vclaim/` directory containing JSON files with verified claims.
- A **Queries** file containing input claims (iclaims).
- A **Qrels** file defining relevance between iclaims and vclaims.

Below is how to access each dataset and where they are located in this repository:

---

### 📌 CheckThat22 Dataset

The **CheckThat22** dataset is available under a research-use license and can be accessed from the [CheckThat Lab GitLab repository](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab.git).

To download it directly using our pipeline, run:

```bash
./download_clef.sh
```

📁 Stored locally in: `clef2022-checkthat-lab/`

---

### 📌 FactCheckTweet Dataset

The **FactCheckTweet** dataset can be accessed for research purposes via the [University of Michigan LIT Lab](https://lit.eecs.umich.edu/publications.html).

📁 Stored locally in: `fact-check_tweet_dataset/`

---

### 📌 OOD Dataset

The **Out-of-Domain (OOD)** dataset is provided by [Meedan](https://meedan.com/research), compiled from fact-checking organizations running tiplines on WhatsApp. We use the data in accordance with its intended research use. ⚠️ This dataset will soon be released for research purposes by Meedan. 

---

## 📘 Dataset Format

All datasets follow a consistent format. Here's a breakdown of the structure and what each file contains:

---

### 🧾 Verified Claims (`vclaim/*.json`)

Each verified claim includes the following fields:

```json
{
  "vclaim_id": "unique ID of the verified claim",
  "vclaim": "text of the verified claim",
  "date": "date the claim was verified",
  "truth_label": "truth verdict (true/false)",
  "speaker": "original source of the claim",
  "url": "link to the fact-check article",
  "title": "title of the fact-check article",
  "text": "full text of the article"
}
```

**Example:**

```json
{
  "vclaim_id": 0,
  "vclaim": "Schuyler VanValkenburg co-sponsored a bill that would have \"allowed abortion until the moment of birth.\"",
  "url": "/factchecks/2019/oct/30/gaydonna-vandergriff/...",
  "speaker": "GayDonna Vandergriff",
  "truth_label": false,
  "date": "stated on October 22, 2019 in a campaign mailer.",
  "title": "Vandergriff inaccurately describes bill that would have eased late-term abortion laws",
  "text": "Republican GayDonna Vandergriff has turned to abortion in her effort to portray ..."
}
```

---

###  Queries File (`queries.txt` or similar)

A **TAB-separated** file with the list of input claims (referred to as `iclaim`):

Format:
```
iclaim_id <TAB> iclaim
```

| iclaim_id      | iclaim                                                                                              |
|----------------|-----------------------------------------------------------------------------------------------------|
| tweet-en-0008  | im screaming. google featured a hoax article that claims Minecraft is being shut down in 2020...    |
| tweet-en-0335  | BREAKING: Footage in Honduras giving cash 2 women & children 2 join the caravan...                 |
| tweet-en-0622  | y’all really joked around so much that tide put their tide pods in plastic boxes…smh               |

---

### 📎 Qrels File (`qrels.txt`)

A **TAB-separated** file that defines which verified claims are relevant to which input claims.

Format:
```
iclaim_id <TAB> 0 <TAB> vclaim_id <TAB> relevance
```

- `0` is a placeholder required by the TREC format.
- `relevance` is `1` if the vclaim proves the iclaim.
- Only pairs with relevance = `1` are listed (relevance = `0` is assumed for all others).

**Example:**

| iclaim_id      | 0 | vclaim_id         | relevance |
|----------------|---|-------------------|-----------|
| tweet-en-0422  | 0 | vclaim-sno-00092  | 1         |
| tweet-en-0538  | 0 | vclaim-sno-00454  | 1         |
| tweet-en-0221  | 0 | vclaim-sno-00012  | 1         |
| tweet-en-0137  | 0 | vclaim-sno-00504  | 1         |


## 🔁 Generating and Verifying Misinformation Edits

This section outlines how we generate and verify misinformation edits using GPT4o. The process is modular and consistent across different perturbation types such as `rewrite`, `negation`, `typos`, `amplify_minimize`, etc.

We use `rewrite` as the example below, but the same process applies for all other edit types.

---

### LLM as a Perturber

Perturbed versions of input claims are generated using an LLM configured via a `config.ini` file located in:

```
experiments/rewrite/gpt4o/config.ini
```

This configuration includes dataset selection, prompt templates, sampling temperature, and model settings.

**Example `config.ini` (for `rewrite`):**

```ini
[data]
dataset = clef2021-checkthat-task2a--english

[model]
model_string = gpt-4o
temperature = 0.9
verification_temperature = 0
prompt_template = "You are now a social media user tasked with rewritting a given tweet in different ways. You will receive two inputs:
    1. A claim (original tweet)  
    2. A fact-check that supports or refutes the claim

    Your task: 
    - Rewrite the tweet in {number_of_samples} different ways while introducing as minimal edits as possible.

    Ensure that:
    - The fact-check is still applicable to the rewritten tweets.
    - The rewritten tweets must convey the same main claim as the original.  
    - The rewritten tweets should read naturally and not appear suspiciously altered.

    **Example**:  
    Original tweet: "Biden signed an executive order today banning the term 'China virus'."  
    Fact-check: "President Joe Biden issued an executive order in January 2021 banning the term ‘China virus.’"  

    Possible rewrites:  
    - "President Biden officially signed an executive order today prohibiting the use of the term 'China virus.'"
    - "Biden's executive order today included a ban on the term 'China virus."
    - "The term 'China virus' has been banned under a new executive order signed by Biden today."
    - "Today, Biden signed an executive order prohibiting the phrase 'China virus."
    - "Biden's executive order banning the term 'China virus' was signed today."

    Response Format:
    Rewritten Tweet 1: [Your first rewritten version]
    Rewritten Tweet 2: [Your second rewritten version]
    Rewritten Tweet 3: [Your third rewritten version]
    Rewritten Tweet 4: [Your fourth rewritten version]
    Rewritten Tweet 5: [Your fifth rewritten version]

    Inputs:
    Tweet: {claim}
    Fact Check: {fact_check}
    Generate your response in the specified string format."

verification_prompt_template = "You are now a fact checker tasked with verifying whether a fact check is applicable to a list of rewritten tweets. You will 
    receive three inputs:
    1. Fact-check: A statement supporting or refuting a claim.
    2. Original Tweet: The source tweet conveying the claim.
    3. Rewritten Tweets: A list of tweets rewritten based on the original tweet.

    Your task: For each rewritten tweet, evaluate:
    - Does the fact-check apply to it?
    - Does it convey the same main claim as the original tweet?
    - Does the rewritten tweet read naturally?

    Your output:
    - For each rewritten tweet, provide a binary label (1 or 0) indicating whether the constraints above are satisfied.

    **Example**:
    Fact-check: "President Joe Biden issued an executive order in January 2021 banning the term ‘China virus.’"
    Original tweet: "Biden signed an executive order today banning the term 'China virus'."  

    Rewritten tweets:  
    - "President Biden officially signed an executive order today prohibiting the use of the term 'China virus.'"
    - "Biden's executive order today included a ban on the term 'China virus."
    - "The term 'China virus' has been banned under a new executive order signed by Biden today."
    - "Today, Biden signed an executive order prohibiting the phrase 'China virus."
    - "Biden's executive order banning the term 'Bejing virus' was signed today."

    Output:
    {
        "labels": [1, 1, 1, 0]
    }

    Inputs:
    Fact Check: {fact_check}
    Original Tweet: {claim}
    Rewritten Tweets: {rewrites}
    Generate your response in the specified JSON format."

[generation]
number_of_samples = 5
```

### 📊 Evaluation Setup

Finally, to evaluate robustness, the config includes an evaluation section:

```ini
[evaluation]
data_directory = <path to cache all the embeddings outputs from evaluations>
embedding_models = all-mpnet-base-v2,all-MiniLM-L12-v2,sentence-t5-base,sentence-t5-large,all-distilroberta-v1,hkunlp/instructor-base,hkunlp/instructor-large,nvidia/NV-Embed-v2,Salesforce/SFR-Embedding-Mistral,sentence-t5-large-ft,all-mpnet-base-v2-ft
original_baseline_path = orig_baseline_rewrite.tsv
edited_baseline_path = edited_baseline_rewrite.tsv
original_worstcase_path = orig_worstcase_rewrite.tsv
edited_worstcase_path = edited_worstcase_rewrite.tsv
```
---

> 📝 To use other perturbation types (e.g., `negation`, `typos`, etc.), replicate the same structure under `experiments/<edit_name>/` and adjust the config accordingly.

--- 

## 🚀 Example Usage: Generating Rewrites

To generate misinformation edits (e.g., rewrites), use the `generate-rewrite` alias with the following arguments:

```bash
generate-rewrite <config_directory> <dataset_name>
```

- `config_directory`: Folder containing `config.ini`
- `dataset_name`: Either `fact-check-tweet` or `clef2021-checkthat-task2a--english`

### ✅ Example

```bash
/claim-matching-robustness$ generate-rewrite experiments/rewrite/gpt4o/ fact-check-tweet
```

This loads the config and generates rewrites using ``GPT4o`` and the specified dataset.

### 📁 Output

The output is saved to:

```
experiments/rewrite/gpt4o/fact-check-tweet/llm_rewrites.jsonl
```

Each line in the file is a JSON object representing a rewritten claim.

---

## 🚀 Example Usage: Verifying Rewrites

To verify the generated misinformation edits (e.g., rewrites), use the `verify-rewrite` alias with the following arguments:

```bash
verify-rewrite <config_directory> <dataset_name>
```

- `config_directory`: Folder containing `config.ini`
- `dataset_name`: Either `fact-check-tweet` or `clef2021-checkthat-task2a--english`

### ✅ Example

```bash
/claim-matching-robustness$ verify-rewrite experiments/rewrite/gpt4o/ fact-check-tweet
```

This command loads the configuration and verifies the rewrites using `GPT-4o` on the specified dataset.

### 📁 Output

The verified rewrites are saved to:

```
experiments/rewrite/gpt4o/fact-check-tweet/llm_rewrites_verified.jsonl
```

Each line in the file is a JSON object representing the verification results.

For a full example, see [experiments/rewrite/gpt4o/clef2021-checkthat-task2a--english/llm_rewrites_verified.jsonl](experiments/rewrite/gpt4o/clef2021-checkthat-task2a--english/llm_rewrites_verified.jsonl).

> ⚠️ Make sure you’ve already generated rewrites before running verification. \
> ⚠️ Ensure your `.env.local` file includes `OPENAI_API_KEY`, and load it into the session with:
> 
> ```bash
> source .env.local
> ```

---

## 🚀 Example Usage: Selecting Rewrites

To select the verified misinformation edits (e.g., rewrites), use the `select-rewrite` alias with the following arguments:

```bash
select-rewrite <config_directory> <dataset_name>
```

- `config_directory`: Folder containing `config.ini`
- `dataset_name`: Either `fact-check-tweet` or `clef2021-checkthat-task2a--english`

### ✅ Example

```bash
/claim-matching-robustness$ select-rewrite experiments/rewrite/gpt4o/ fact-check-tweet
```

This command loads the configuration and selects verified rewrites by applying logic to identify both:
- **Baseline rewrites** (least changed)
- **Worst-case rewrites** (most significantly altered)

### 📁 Output

The selected rewrites are saved to four files:

- `original_baseline_rewrite.tsv` — Original input claims linked to the baseline edits  
- `edited_baseline_rewrite.tsv` — Baseline rewrites with minimal edits  
- `original_worstcase_rewrite.tsv` — Original input claims linked to the worst-case edits  
- `edited_worstcase_rewrite.tsv` — Worst-case rewrites 

All output files are saved under:

```
experiments/rewrite/gpt4o/<dataset_name>/
```

> ⚠️ Make sure verification is complete before running selection.

# Retrievers Evaluation (Before Reranking)

This section explains how to evaluate different embedding models before reranking. All embedding models are used off-the-shelf with default hyperparameters, using the [SentenceTransformers library](https://sbert.net/).

Use the following flags with the scripts:

- `--save-embs`: Saves verified claim embeddings to speed up future runs.
- `--save-ranks`: Saves rankings from first-stage retrieval for use in reranking.

### clef2021-checkthat-task2a--english

Run the following command to evaluate retrievers on the `clef2021-checkthat-task2a--english` dataset:

```bash
python src/claimrobustness/evaluate/before_reranking_all.py clef2021-checkthat-task2a--english --save-embs --save-ranks
``` 

### fact-check-tweet
The ``fact-check-tweet`` dataset contains longer articles, and many embedding models support only up to 512 tokens. For this reason, articles are split into paragraphs, and similarity is computed between each paragraph and the input text.

Run the following script:

```bash
python src/claimrobustness/evaluate/before_reranking_paragraph_all.py fact-check-tweet --save-embs --save-ranks
``` 

## BM25 Evaluation
BM25 evaluation is implemented separately. To run BM25 on a specific dataset, use the script below:

```bash
python src/claimrobustness/evaluate/before_reranking_bm25_all.py clef2021-checkthat-task2a--english
```

## Running evaluations on Dialect 
Since dialect transformations differ from other misinformation edits, they use different scripts for evaluation.

### Without splitting paragraphs - clef2021-checkthat-task2a--english

```bash
python src/claimrobustness/evaluate/dialect_ranking.py clef2021-checkthat-task2a--english --save-embs --save-ranks
```

### With paragraph splitting
Use this script for datasets like ``fact-check-tweet`` that require splitting into paragraphs:

```bash
python src/claimrobustness/evaluate/dialect_ranking_paragraph.py fact-check-tweet --save-embs --save-ranks
```

### BM25 on Dialect Edits
To run BM25 retrieval on dialect edits:

```bash 
python src/claimrobustness/evaluate/dialect_bm25.py clef2021-checkthat-task2a--english --save-ranks
```

Note: Run all scripts from the root of the project directory. Adjust paths if running from another location. The scripts above output results to:

> ```experiments/<edit_type>/<dataset>/before_reranking_results_all.jsonl```

## Reranker Evaluation (After Reranking)

To evaluate different reranker models after running the retrieval step above, run the script below. We use a default of 50 candidates but this can be changed for different experimentation. List of experiments supported - 

```bash
python src/claimrobustness/evaluate/reranker.py <dataset> --n-candidates=50 --model-name=<reranker> --experiments [list of experiments] --include-bm25
``` 

### Example Usage:
```
python src/claimrobustness/evaluate/reranker.py fact-check-tweet --n-candidates=50 --model-name=bge_llm --experiments casing rewrite negation --include-bm25
```

This outputs a jsonl file in ```experiments/<edit_type>/<dataset>/before_reranking_results_all.jsonl``` 

## Mitigation
This section contains code and instructions for running the mitigation approaches. All the code and datasets can be found in this folder  📁 `mitigation/`

### Knowledge Distillation Approach

We use the same perturbation generation framework to create perturbed claims, which are then paired for training.

- The **full training set** (70,954 claim pairs) is located at:  
  ```mitigation/train_perturbed_queries_full_train.csv```

- The **lite training set** (11,593 claim pairs) is located at:  
  ```mitigation/train_perturbed_queries_lite_train.csv```

To apply the knowledge distillation approach, we adapt the code from [`model_distillation.py`](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/training/distillation/model_distillation.py), based on the paper [*Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*](https://arxiv.org/abs/2004.09813).

Our implementation can be found in:  
```mitigation/make_robust.py```

To run the knowledge distillation approach, use:

```bash
python src/claimrobustness/mitigation/make_robust.py
``` 

This runs KD on the specified embedding model and outputs the robust model to the specified output file. The resulting model can then be evaluated by updating the list of embedding models in the evaluation scripts to include the newly generated model.

### Claim Normalization Approach 
To run claim normalization (CN), use the command below specifying the folder that contains the config.ini and the dataset name.

Example [config.ini](experiments/mitigation/gpt4o/config.ini)

```ini
[model]
model_string = gpt-4o
prompt_template = "You will be provided with a noisy input claim from a social media post. 
    The input claim may contain informal language, typos, abbreviations, double negations and dialectal variations.
    Your task is to normalise the claim to a more formal and standardised version while preserving the original meaning.

    Ensure that:
    - The normalised claim conveys the same main claim as the original.

    Let's see an example:
    Noisy Claim: "Wah, Biden just sign order today, cannot call ‘China virus’ liao"
    Normalised Claim: "President Joe Biden issued an executive order today banning the term ‘China virus.’"

    Noisy Claim: "Soros son sez he and dad pickd Harris 4 VP after pic interview!"
    Normalised Claim: "George Soros son revealed that he and his father chose Kamala Harris as the Vice President after a picture interview."

    Noisy Claim: "It is not untrue that President-elect Joe Biden’s German shepherd, Major, is set to become the first shelter dog in the White House."
    Normalised Claim: "President-elect Joe Biden’s German shepherd, Major, is set to become the first shelter dog in the White House."

    Response Format:
    Normalised Claim: [Your normalised claim]

    Inputs:
    Noisy Claim: {claim}
    Generate your response in the specified string format."
```

You can specify the dataset to run CN on by editing the dict at the start of the file.
```python 
data_paths = {
    # "typos": "experiments/typos/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_typos.tsv",
    # "entity_replacement": "experiments/named_entity_replacement/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_named_entity_replacements.tsv",
    "dialect_pidgin": "experiments/dialect/gpt4o/clef2021-checkthat-task2a--english/pidgin/orig_baseline_dialect.tsv",
    # "negation": "experiments/negation/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_negation_queries.tsv",
}
```  

Example Usage:
```bash
python src/claimrobustness/mitigation/generator.py experiments/mitigation/gpt4o/ clef2021-checkthat-task2a--english
```

## Analyse Results
To analyse and visualise the results, we provide this [notebook](notebooks/analyse_results.ipynb) that we use for analysis. 


## BibTeX
If you find our work useful, please consider citing our paper!

```bibtex
@misc{magomere2025claimsevolveevaluatingenhancing,
      title={When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits}, 
      author={Jabez Magomere and Emanuele La Malfa and Manuel Tonneau and Ashkan Kazemi and Scott Hale},
      year={2025},
      eprint={2503.03417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03417}, 
}
```