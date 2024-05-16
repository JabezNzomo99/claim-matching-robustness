# Evaluating robustness of claim matching methdods to misinformation edits

## Data
To load the data used in the experiments, run the script below to download the data from CLEFCheckThat22 edition.

```
chmod +x download_clef.sh
.\download_clef.sh
```

## Misinformation Edits
### Named Entity Replacements
To generate named entity replacement edits on input claims, set up the following config and specify the different parameters as shown below.

```
[data]
dataset = clef2021-checkthat-task2a--english

[model]
model_string = llama3-70b-8192
temperature = 0.9
prompt_template = "You will receive three inputs:
    1. A claim made on social media
    2. A fact-check of that claim
    3. A list of named entities (people, places, organizations, etc.) to replace in the original claim
    
    Your task is to generate a list of {number_of_samples} alternative name entities that can replace the each original one in the claim, while making sure the fact-check still applies to the modified claim
    and the modified claim is grammatically correct. 
    
    Return your output as a JSON object containing the list of swappable name entity tokens.

    Claim: {claim}
    Fact Check: {fact_check}
    Named Entity Tokens: {named_entities}

    Response format: {{
        "replaceable_entities": [
            {{
                "token": string,
                "replacements": [string]
            }}
        ]
    }}"

[generation]
number_of_samples = 3
baseline = 1
worstcase = 3

[verifier]
model_string = microsoft/deberta-v3-base
model_path = /experiments/train_verifier/debertaV3/
num_labels = 2
```

#### Usage
```
usage: generate.py [-h] [--no-baseline] [--no-worstcase] experiment_path

positional arguments:
  experiment_path  path where config lies

options:
  -h, --help       show this help message and exit
  --no-baseline    Skip generating baseline edits
  --no-worstcase   Skip generating worstcase edits
```

#### Example
