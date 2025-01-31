import logging
import random
import string
import torch
import os
from collections import Counter
from tqdm import tqdm
import gc

from c2nl.objects import Code, Summary
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, CLS_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer

logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(lang_id,
                     source,
                     source_tag,
                     target,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True,
                     mean_latent=False):
    tokenizers = RobertaTokenizer.from_pretrained('Salesforce/codet5p-220m')
    code_tokens = tokenizers.tokenize(source)
    code_type = []
    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            return None

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    if len(code_tokens) == 0:
        return None

    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP
    code = Code()
    code.text = source
    code.language = lang_id
    code.tokens = code_tokens
    if mean_latent:
        code.prepend_token(CLS_WORD)
    code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
    if code_tag_type != 'subtoken':
        code.mask = [1 if ct == 'N' else 0 for ct in code_type]

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = tokenizers.tokenize(summ)
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None

    example = dict()
    example['code'] = code
    example['summary'] = summary
    return example


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src']) as f:
        sources = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['src']))]

    if filenames['tgt'] is not None:
        with open(filenames['tgt']) as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(sources)

    if filenames['src_tag'] is not None:
        with open(filenames['src_tag']) as f:
            source_tags = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_tag']))]
    else:
        source_tags = [None] * len(sources)

    assert len(sources) == len(source_tags) == len(targets)

    # Construct the filename to save/load the code_cls tensor
    dir_name = os.path.dirname(filenames['src'])
    base_name = os.path.basename(filenames['src'])
    new_filename = os.path.join(dir_name, base_name + "_cls.pt")

    model_path = os.path.join(os.path.dirname(dir_name), "checkpoint-best-bleu")

    # Check if the file already exists
    if os.path.exists(new_filename):
        code_cls = torch.load(new_filename, map_location=torch.device('cpu'))
        print(f"Loaded code_cls tensor from {new_filename}")

    else:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        codeBert = AutoModel.from_pretrained("microsoft/codebert-base").cuda()

        # Define a function to split the data into batches
        def batchify(data, batch_size):
            return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # Define batch size, adjust based on your needs
        BATCH_SIZE = 129

        # Split sources into batches
        source_batches = batchify(sources, BATCH_SIZE)

        # Placeholder to collect the [CLS] hidden states for all batches
        code_cls_list = []

        # Process each batch with a progress bar
        for batch in tqdm(source_batches, desc="Processing batches"):

            # Determine the max length for this batch (not exceeding 512)
            max_length_for_this_batch = min(max(len(tokenizer.tokenize(text)) for text in batch), 512)

            encoded_inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='max_length',
                max_length=max_length_for_this_batch,
                truncation=True
            )

            # Move encoded inputs to GPU
            encoded_inputs = {key: value.cuda(non_blocking=True) for key, value in encoded_inputs.items()}

            with torch.no_grad():
                outputs = codeBert(**encoded_inputs)

            # Extract the [CLS] hidden states for the current batch, detach from the graph, and append to the list
            code_cls_list.append(outputs.last_hidden_state[:, 0, :].detach().cpu())

            #remove temp variables from memory
            del encoded_inputs
            del outputs
            torch.cuda.empty_cache()

        # Convert list of tensors to a single tensor on CPU
        code_cls = torch.cat(code_cls_list, dim=0).cpu().unsqueeze(1)

        # Save the tensor to the file
        torch.save(code_cls, new_filename)
        print(f"Saved code_cls tensor to {new_filename}")
        del tokenizer
        del codeBert
        gc.collect()
        torch.cuda.empty_cache()

    examples = []
    for src, src_tag, tgt, cls in tqdm(zip(sources, source_tags, targets, code_cls),
                                  total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   src,
                                   src_tag,
                                   tgt,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split,
                                   mean_latent=args.mean_latent)
            if _ex is not None:
                _ex['code'].code_cls = cls.clone()
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, CLS_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].tokens)

    # -3 to reserve spots for PAD UNK and CLS token
    dict_size = dict_size - 3 if dict_size and dict_size > 3 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictionary = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictionary


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
