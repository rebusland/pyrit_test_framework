from pyrit.score import (
    AzureContentFilterScorer,
    FloatScaleThresholdScorer,
    SubStringScorer,
    SelfAskRefusalScorer,
    SelfAskLikertScorer,
    LikertScalePaths
)
import os

def get_float_threshold_azure_content_filter_scorer(api_key, full_endpoint, threshold):
    azure_content_filter_scorer = AzureContentFilterScorer(
        endpoint=full_endpoint,
        api_key=api_key,
    )
    return FloatScaleThresholdScorer(scorer=azure_content_filter_scorer, threshold=threshold)

def get_substring_scorer(substring: str, category: str=None):
    return SubStringScorer(
        substring=substring,
        category=category
    )

def get_self_ask_refusal_scorer(target_checking_refusal):
    return SelfAskRefusalScorer(chat_target=target_checking_refusal)

# TODO HOW TO SET SCALE
def get_self_ask_likert_scorer(target, likert_scale_path=LikertScalePaths.SEXUAL_SCALE.value):
    return SelfAskLikertScorer(chat_target=target, likert_scale_path=likert_scale_path)

def get_composite_and_scorer():
    pass
