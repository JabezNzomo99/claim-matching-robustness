# paths for clef checkthat task 2a english
TASK_2A_EN_PATH = "clef2022-checkthat-lab/task2/data/subtask-2a--english"

TASK_2A_EN_TRAIN_QUERY_PATH = (
    TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train-Dev_Queries.tsv"
)
TASK_2A_EN_DEV_QUERY_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train-Dev_Queries.tsv"
TASK_2A_EN_TEST21_QUERY_PATH = (
    TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev-Test_Queries.tsv"
)
TASK_2A_EN_TEST22_QUERY_PATH = (
    TASK_2A_EN_PATH + "/test/CT2022-Task2A-EN-Test_Queries.tsv"
)

TASK_2A_EN_TARGETS_PATH = TASK_2A_EN_PATH + "/vclaims"
TASK_2A_EN_TARGETS_KEY_NAMES = [
    "title",
    "subtitle",
    "author",
    "date",
    "target_id",
    "target",
    "page_url",
]

TASK_2A_EN_TRAIN_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train_QRELs.tsv"
TASK_2A_EN_DEV_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev_QRELs.tsv"
TASK_2A_EN_TEST21_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev-Test_QRELs.tsv"

FC_DATASET_PATH = "/home/kebl7383/claim-matching-robustness/fact-check_tweet_dataset"  # TODO: Change this to the correct path
FC_EN_TRAIN_QUERY_PATH = FC_DATASET_PATH + "/FC-EN-Train-Queries.tsv"
FC_EN_DEV_QUERY_PATH = FC_DATASET_PATH + "/FC-EN-Train-Queries.tsv"
FC_EN_TEST_QUERY_PATH = FC_DATASET_PATH + "/FC-EN-Test-Queries.tsv"
FC_TARGETS_PATH = FC_DATASET_PATH + "/vclaims"
FC_EN_TRAIN_QREL_PATH = FC_DATASET_PATH + "/FC-EN-Train_QRELs.tsv"
FC_EN_DEV_QREL_PATH = FC_DATASET_PATH + "/FC-EN-Train_QRELs.tsv"
FC_EN_TEST_QREL_PATH = FC_DATASET_PATH + "/FC-EN-Test_QRELs.tsv"
FC_TARGETS_KEY_NAMES = [
    "title",
    "subtitle",
    "author",
    "date",
    "target_id",
    "url",
    "target",
]

# Create a dictionary with the types - manually
NAMED_ENTITIES_TYPE_DEFINITIONS = {
    "PERSON": "People, including fictional",
    "NORP": "Nationalities or religious or political groups",
    "FAC": "Buildings, airports, highways, bridges, etc.",
    "ORG": "Companies, agencies, institutions, etc.",
    "GPE": "Countries, cities, states",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water",
    "PRODUCT": "Vehicles, weapons, foods, etc. (Not services)",
    "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
    "WORK_OF_ART": "Titles of books, songs, etc. ",
    "LAW": "Named documents made into laws ",
    "LANGUAGE": "Any named language ",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "PERCENT": "Percentage (including “%”)",
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements, as of weight or distance ",
    "ORDINAL": "“first”, “second”",
    "CARDINAL": "Numerals that do not fall under another type",
}

SEPARATOR_TOKEN = " [SEP] "
