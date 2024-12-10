import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

########################################################################
################## For Encoder #########################################
########################################################################

class ClaimsEncodingDataset(Dataset):
    def __init__(self, tokenized_claims):
        self.data = tokenized_claims

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tclaim = self.data[index]
        return np.array(tclaim["input_ids"]), np.array(tclaim["attention_mask"])

def get_encoding_dataloader(tokenized_claims, params=dict(batch_size=16, shuffle=False)):
    dataset = ClaimsEncodingDataset(tokenized_claims)
    return DataLoader(dataset, **params)


########################################################################
################## For an extended GPT #################################
########################################################################

def square_mask(mask):
    return np.einsum("i,j->ij", mask, mask)

def cat_masks(masks):
    assert len(masks) > 0
    seq_lens = [mask.shape[1] for mask in masks]
    total_seq_len = sum(seq_lens)
    total_mask = np.zeros((total_seq_len, total_seq_len))
    idx = 0
    for seq_len, mask in zip(seq_lens, masks):
        total_mask[idx:idx+seq_len, idx:idx+seq_len] = mask
        idx += seq_len
    return total_mask  

class ExtendedAutoRegressiveDataset(Dataset):
    def __init__(self, 
        tweet_encode_fn, 
        claim_encode_fn, 
        claims, 
        tweets, 
        connections, 
        ranks=None,
        prior_prob=0,
        mask_prior=False,
        include_posterior=False,
        training="mle",
        ):
        """Dataset for training AR model
        Args:
            tweet_encode_fn: fn to encode tweet
            claim_encode_fn: fn to encode claim
            claims: claim df
            tweets: tweets df
            connections: contains the pairings
            ranks: negative rankings (should not include gold label)
        """
        self.claims = claims
        self.claims["encoded_vclaim"] = self.claims.target.apply(claim_encode_fn)
        self.tweets = tweets
        self.tweets["encoded_tweet"] = self.tweets['query'].apply(tweet_encode_fn)

        run_tweets = self.tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(self.claims, on="target_id", how="inner")
        run_tweets = run_tweets[["encoded_tweet", "encoded_vclaim"]].reset_index()

        self.data = run_tweets
        self.ranks = ranks
        self.prior_prob = prior_prob
        self.mask_prior = mask_prior
        self.include_posterior = include_posterior
        self.training = training
        self.mixed = training == "mixed"

        assert prior_prob == 0 or not mask_prior
        assert prior_prob == 0 or not self.include_posterior
        assert mask_prior == False or not self.include_posterior
        assert not (self.mixed and self.ranks is None)

    def __len__(self):
        return len(self.data)
    
    def _split_index(self, index):
        i1 = index % len(self.data)
        i2 = index >= len(self.data)
        return i1, i2

    def _prior_only_params(self, index):
        """When only training priors, efficiently stack 2 via masking
        """
        claim_index = np.random.randint(0, len(self.claims))
        tweet_index = np.random.randint(0, len(self.tweets))

        left = self.claims["encoded_vclaim"][claim_index]
        right = self.tweets["encoded_tweet"][tweet_index]
        
        attn_mask_left = square_mask(np.array(left["attention_mask"]))
        attn_mask_right = square_mask(np.array(right["attention_mask"]))
        attn_mask = cat_masks([attn_mask_left, attn_mask_right]).tolist()
        
        token_ids = left["input_ids"] + right["input_ids"]
        labels = [iid if mid else -100 for iid, mid in zip(token_ids, left["attention_mask"]+right["attention_mask"])]

        position_ids = list(range(len(left["input_ids"]))) + list(range(len(right["input_ids"])))
        return token_ids, attn_mask, position_ids, labels

    def _joint_params(self, index, overwrite_claim_index=None, posterior=False):
        """returns inputs for p(x), p(y|x)
        """
        if overwrite_claim_index is None:
            left = self.data.encoded_vclaim[index]
        else:
            left = self.claims.encoded_vclaim[overwrite_claim_index]
        right = self.data.encoded_tweet[index]

        if posterior:
            tmp = left
            left = right
            right = tmp

        attn_mask = left["attention_mask"] + right["attention_mask"]
        attn_mask = square_mask(np.array(attn_mask)).tolist()

        token_ids = left["input_ids"] + right["input_ids"]
        if not self.mask_prior:
            labels = [iid if mid else -100 for iid, mid in zip(token_ids, left["attention_mask"]+right["attention_mask"])]
        else:
            labels = [-100] * len(left["attention_mask"]) + \
                [iid if mid else -100 for iid, mid in zip(right["input_ids"], right["attention_mask"])]

        position_ids = list(range(len(left["input_ids"]))) + list(range(len(right["input_ids"])))
        return token_ids, attn_mask, position_ids, labels

    def _with_negative_params(self, index):
        """Each sample comes with a corresponding negative
        """
        rank_index = 0#int(np.random.exponential(3))
        negative_index = self.ranks[index][rank_index]

        positive_params = self._joint_params(index)
        negative_params = self._joint_params(index, overwrite_claim_index=negative_index)

        return list(zip(positive_params, negative_params))

    def _with_posterior_params(self, index):
        """get x|y and y|x
        """
        likelihood_params = self._joint_params(index)
        posterior_params = self._joint_params(index, posterior=True)

        return list(zip(likelihood_params, posterior_params))

    def _mixed_params(self, index):
        rank_index = int(np.random.exponential(3))
        negative_index = self.ranks[index][rank_index]

        likelihood_params = self._joint_params(index)
        posterior_params = self._joint_params(index, posterior=True)
        negative_params = self._joint_params(index, overwrite_claim_index=negative_index)

        return list(zip(likelihood_params, posterior_params, negative_params))


    def __getitem__(self, index):
        if self.mixed:
            token_ids, attn_mask, position_ids, labels = self._mixed_params(index)

        elif self.ranks is not None:
            token_ids, attn_mask, position_ids, labels = self._with_negative_params(index)

        elif self.include_posterior:
            token_ids, attn_mask, position_ids, labels = self._with_posterior_params(index)
        
        elif self.prior_prob > 0 and np.random.binomial(1, self.prior_prob):
            token_ids, attn_mask, position_ids, labels = self._prior_only_params(index)

        else:
            token_ids, attn_mask, position_ids, labels = self._joint_params(index)

        return {
            "input_ids": token_ids, 
            "attention_mask": attn_mask, 
            "labels": labels,
            "position_ids": position_ids
        }

########################################################################
################## For an extended GPT Eval ############################
########################################################################

def verified_array_cat(list1, list2):
    return np.concatenate([
        np.array(list1), 
        np.array(list2)], axis=-1)

class ExtendedAutoRegressiveEvalDataset(Dataset):
    def __init__(self, tweet_encode_fn, claim_encode_fn, claims, tweets, connections, ranks, n_candidates=5):
        self.claims = claims
        self.claims["encoded_vclaim"] = self.claims.target.apply(claim_encode_fn)

        run_tweets = tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(claims, on="target_id", how="inner")
        run_tweets = run_tweets[["query", "encoded_vclaim"]].reset_index()
        run_tweets["encoded_tweet"] = run_tweets['query'].apply(tweet_encode_fn)
        # run_tweets["encoded_vclaim"] = run_tweets.tweet.apply(claim_encode_fn)
        self.data = run_tweets
        self.ranks = ranks
        self.n_candidates = n_candidates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        enc_tweet = self.data.encoded_tweet[index]
        enc_claim_pos = self.data.encoded_vclaim[index]
        enc_claim_negs = self.claims.encoded_vclaim[self.ranks[index][:self.n_candidates]]

        token_ids_pos = verified_array_cat(enc_claim_pos["input_ids"], enc_tweet["input_ids"])
        token_ids_negs = [verified_array_cat(enc_claim_neg["input_ids"], enc_tweet["input_ids"]) \
            for enc_claim_neg in enc_claim_negs]
        attn_mask_pos = verified_array_cat(enc_claim_pos["attention_mask"], enc_tweet["attention_mask"])
        attn_mask_negs = [verified_array_cat(enc_claim_neg["attention_mask"], enc_tweet["attention_mask"]) \
            for enc_claim_neg in enc_claim_negs]
        
        label_mask = verified_array_cat(np.zeros((len(enc_claim_pos["attention_mask"]),)), enc_tweet["attention_mask"])
        labels = token_ids_pos * label_mask + -100 * (1 - label_mask)
        labels = labels.astype(token_ids_pos.dtype)

        position_ids = verified_array_cat(np.arange(len(enc_claim_pos["input_ids"])), np.arange(len(enc_tweet["input_ids"])))

        return {
            "input_ids": token_ids_negs,
            "attention_mask": attn_mask_negs,
            "labels": labels,
            "position_ids": position_ids
        }

class ExtendedAutoRegressiveScoringDataset(Dataset):
    def __init__(self, tweet_encode_fn, claim_encode_fn, claims, tweets, ranks, n_candidates=5):
        self.claims = claims
        self.claims["encoded_vclaim"] = self.claims.vclaim.apply(claim_encode_fn)
        self.data = tweets
        self.data["encoded_tweet"] = self.data.tweet.apply(tweet_encode_fn)
        self.ranks = ranks
        self.n_candidates = n_candidates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        enc_tweet = self.data.encoded_tweet[index]
        enc_claims = self.claims.encoded_vclaim[self.ranks[index][:self.n_candidates]]

        token_ids = [verified_array_cat(enc_claim["input_ids"], enc_tweet["input_ids"]) \
            for enc_claim in enc_claims]
        attn_masks = [verified_array_cat(enc_claim["attention_mask"], enc_tweet["attention_mask"]) \
            for enc_claim in enc_claims]
        
        # print(type(enc_claims), enc_claims)
        label_mask = verified_array_cat(np.zeros((len(enc_claims.iloc[0]["attention_mask"]),)), enc_tweet["attention_mask"])
        labels = token_ids[0] * label_mask + -100 * (1 - label_mask)
        labels = labels.astype(token_ids[0].dtype)

        position_ids = verified_array_cat(np.arange(len(enc_claims.iloc[0]["input_ids"])), np.arange(len(enc_tweet["input_ids"])))

        return {
            "input_ids": token_ids,
            "attention_mask": attn_masks,
            "labels": labels,
            "position_ids": position_ids
        }