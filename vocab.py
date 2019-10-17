from collections import Counter
from itertools import chain
import logging

from utils import write_json_data, read_json_data


logger = logging.getLogger(__name__)


class VocabEntry:
    MAX_SENT_LEN = 250

    def __init__(self, word2id=None, max_sent_len=0):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id["<pad>"] = 0
            self.word2id["<s>"] = 1
            self.word2id["</s>"] = 2
            self.word2id["<unk>"] = 3
        self.unk_id = self.word2id["<unk>"]
        self.id2word = {str(v): k for k, v in self.word2id.items()}
        self.max_sent_len = max_sent_len

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return "Vocabulary[size=%d]" % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[str(wid)]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[str(wid)] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int]
        or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[str(w_id)] for w_id in word_ids]

    def listindices2words(self, list_indices):
        return [self.indices2words(indices) for indices in list_indices]

    def padd_sent(self, sent, start_end=True):
        trun_sent = sent[: self.max_sent_len]
        seq_ids = []
        for word in trun_sent:
            seq_ids.append(self[word])
        padd = (self.max_sent_len - len(seq_ids)) * [self["<pad>"]]
        if start_end:
            seq_ids = [self["<s>"]] + seq_ids + padd
        else:
            seq_ids = seq_ids + padd
        return seq_ids

    def padd_sents(self, sents, start_end=True):
        return [self.padd_sent(sent, start_end) for sent in sents]

    def is_unknown(self, pad_seq_ids):
        val_tokens_id = [self["<s>"], self["</s>"], self["<pad>"], self["<unk>"]]

        for token_id in pad_seq_ids:
            if token_id not in val_tokens_id and str(token_id) in self.id2word:
                return 0
        return 1

    def save_json(self, file_path):
        save_data = {
            "unk_id": self.unk_id,
            "max_sent_len": self.max_sent_len,
            "word2id": self.word2id,
            "id2word": self.id2word,
        }

        write_json_data(file_path, save_data)

    @classmethod
    def from_json(cls, file_path):
        vocab_data = read_json_data(file_path)

        return cls(
            word2id=vocab_data["word2id"],
            max_sent_len=vocab_data["max_sent_len"],
        )

    @staticmethod
    def from_corpus(corpus, size=None, max_sent_len=None, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[list[str]]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry):
            VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()

        if max_sent_len is None:
            vocab_entry.max_sent_len = len(max(corpus, key=lambda sent: len(sent)))
        else:
            vocab_entry.max_sent_len = max_sent_len
        if vocab_entry.max_sent_len > VocabEntry.MAX_SENT_LEN:
            vocab_entry.max_sent_len = VocabEntry.MAX_SENT_LEN

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        logger.info(
            "number of word types: {}, number of word "
            "types w/ frequency >= {}: {}".format(
                len(word_freq), freq_cutoff, len(valid_words)
            )
        )
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        if size is not None:
            top_k_words = top_k_words[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


if __name__ == "__main__":
    pass
