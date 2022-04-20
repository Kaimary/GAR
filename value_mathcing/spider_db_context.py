import re
from collections import Set, defaultdict
from typing import Dict, Tuple, List
from nltk.stem import WordNetLemmatizer
from allennlp.data import Tokenizer, Token
from ordered_set import OrderedSet
from unidecode import unidecode
from spider_utils.utils import TableColumn, read_dataset_schema, read_dataset_values
from allennlp_semparse.common.knowledge_graph import KnowledgeGraph

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = {"", "being", "-", "over", "through", "yourselves", "before", ",", "should",
              "cannot", "during", "yourself", "because", "doing", "further", "ourselves",
              "what", "between", "mustn", "?", "shouldn", "couldn", "could", "against",
              "while", "whom", "your", "their", "aren", "there", "wouldn", "themselves",
              ":", "himself", "herself", "haven", "those", "myself", "these", ";", "below",
              "theirs", "doesn", "itself", "!", "again", "that", "when", "which", "yours",
              "having", "and", "\'", ".", "\"", "won", "last", "in", "id", "a", "an", "are",
              "is", "do", "to", "by", "d", "as", "option", "at", "city", "independent"}


def is_number(s):
    """
    detect whether a string is a number
    @param s: string
    @return: Boolean
    """
    try:
        float(s)
        return True
    except (TypeError, ValueError):  # TODO: Find where cause the NoneTypeError
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


class SpiderDBContext:

    # global_neighbors: Dict[str, OrderedSet[str]] = defaultdict(OrderedSet) 
    # global_entities = set()
    # global_entity_text = {}
    # foreign_keys_to_column = {}

    def __init__(self, db_id: str, utterance: str, tokenizer: Tokenizer, tables_file: str, dataset_path: str):
        self.schemas = {}
        self.db_tables_data = {}
        self.string_column_mapping = defaultdict(set)
        self.global_neighbors: Dict[str, OrderedSet[str]] = defaultdict(OrderedSet)
        self.global_entities = set()
        self.global_entity_text = {}
        self.foreign_keys_to_column = {}

        self.dataset_path = dataset_path
        self.tables_file = tables_file
        self.db_id = db_id
        self.tokenizer = tokenizer
        self.utterance = utterance
        self.lemmatizer = WordNetLemmatizer()

        tokenized_utterance = tokenizer.tokenize(utterance.lower())
        self.tokenized_utterance = [Token(text=t.text) for t in tokenized_utterance]

        if db_id not in self.schemas:
            self.schemas = read_dataset_schema(self.tables_file)
        self.schema = self.schemas[db_id]

        self.knowledge_graph = self.get_db_knowledge_graph(db_id)

        entity_texts = [self.knowledge_graph.entity_text[entity].lower()
                        for entity in self.knowledge_graph.entities]
        self.entity_tokens = [[Token(text=t.text) for t in tokenizer.tokenize(entity_text)] for entity_text in
                              entity_texts]
        # self.entity_tokens = []
        # for et in entity_tokens:
        #     tokens = []
        #     for t in et:
        #         t_list = [tt.text for tt in t]
        #         tokens.append(Token(text=''.join(t_list)))
        #     self.entity_tokens.append(tokens)

    def change_utterance(self, utterance: str):
        self.utterance = utterance

        tokenized_utterance = self.tokenizer.tokenize(utterance.lower())
        self.tokenized_utterance = [Token(text=t.text) for t in tokenized_utterance]

    @staticmethod
    def entity_key_for_column(table_name: str, column: TableColumn) -> str:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        return f"column:{column_type.lower()}:{table_name.lower()}:{column.name.lower()}"

    def get_db_knowledge_graph(self, db_id: str) -> KnowledgeGraph:
        entities: Set[str] = set()
        neighbors: Dict[str, OrderedSet[str]] = defaultdict(OrderedSet)
        entity_text: Dict[str, str] = {}
        # foreign_keys_to_column: Dict[str, str] = {}

        db_schema = self.schema
        tables = db_schema.values()

        if db_id not in self.db_tables_data:
            self.db_tables_data[db_id] = read_dataset_values(db_id, self.dataset_path, tables)

        tables_data = self.db_tables_data[db_id]

        # string_column_mapping: Dict[str, set] = defaultdict(set)
        if not self.string_column_mapping:
            for table, table_data in tables_data.items():
                for table_row in table_data:
                    for column, cell_value in zip(db_schema[table.name].columns, table_row):
                        # Make sure cell_value is not NaN and should be string
                        if cell_value == cell_value and isinstance(cell_value, str):
                            cell_value_normalized = self.normalize_string(cell_value)
                        elif isinstance(cell_value, (int, float)):
                            cell_value_normalized = f"{cell_value}"
                        else:
                            cell_value_normalized = self.normalize_string(str(cell_value))

                        column_key = self.entity_key_for_column(table.name, column)
                        self.string_column_mapping[cell_value_normalized].add(column_key)
                        # hard code for major
                        if is_number(cell_value):
                            self.string_column_mapping['major'].add(column_key)

        self.string_entities = self.get_entities_from_question(self.string_column_mapping)

        if not self.global_entities:
            for table in tables:
                table_key = f"table:{table.name.lower()}"
                self.global_entities.add(table_key)
                self.global_entity_text[table_key] = table.text

                for column in db_schema[table.name].columns:
                    entity_key = self.entity_key_for_column(table.name, column)
                    self.global_entities.add(entity_key)
                    self.global_neighbors[entity_key].add(table_key)
                    self.global_neighbors[table_key].add(entity_key)
                    self.global_entity_text[entity_key] = column.text

        for string_entity, column_keys in self.string_entities:
            entities.add(string_entity)
            for column_key in column_keys:
                neighbors[string_entity].add(column_key)
                neighbors[column_key].add(string_entity)
            entity_text[string_entity] = string_entity.split(":")[-1]

        if not self.foreign_keys_to_column:
            # loop again after we have gone through all columns to link foreign keys columns
            for table_name in db_schema.keys():
                for column in db_schema[table_name].columns:
                    if column.foreign_key is None:
                        continue

                    other_column_table, other_column_name = column.foreign_key.split(':')

                    # must have exactly one by design
                    other_column = \
                        [col for col in db_schema[other_column_table].columns if col.name == other_column_name][0]

                    entity_key = self.entity_key_for_column(table_name, column)
                    other_entity_key = self.entity_key_for_column(other_column_table, other_column)

                    self.global_neighbors[entity_key].add(other_entity_key)
                    self.global_neighbors[other_entity_key].add(entity_key)

                    self.foreign_keys_to_column[entity_key] = other_entity_key

        entities.update(self.global_entities)
        entity_text.update(self.global_entity_text)
        neighbors.update(self.global_neighbors)

        kg = KnowledgeGraph(entities, dict(neighbors), entity_text)
        kg.foreign_keys_to_column = self.foreign_keys_to_column

        return kg

    def _string_in_table(self, candidate: str,
                         string_column_mapping: Dict[str, set]):
        """
        Checks if the string occurs in the table, and if it does, returns the names of the columns
        under which it occurs. If it does not, returns an empty list.
        """

        def isfloat(x):
            try:
                a = float(x)
            except (TypeError, ValueError):
                return False
            else:
                return True

        def isint(x):
            try:
                a = float(x)
                b = int(a)
            except (TypeError, ValueError):
                return False
            else:
                return a == b

        candidate_column_names = []
        # First check if the entire candidate occurs as a cell.
        if candidate in string_column_mapping:
            candidate_column_names = string_column_mapping[candidate]
        elif isint(candidate):
            candidate = str(int(candidate) - 10)
            if candidate in string_column_mapping:
                candidate_column_names = string_column_mapping[candidate]
        elif isfloat(candidate):
            candidate = str(float(candidate) - 10)
            if candidate in string_column_mapping:
                candidate_column_names = string_column_mapping[candidate]
        # # If not, check if it is a substring pf any cell value.
        # if not candidate_column_names:
        #     for cell_value, column_names in string_column_mapping.items():
        #         if candidate in cell_value:
        #             candidate_column_names.extend(column_names)
        candidate_column_names = list(set(candidate_column_names))
        return candidate_column_names

    def get_entities_from_question(self,
                                   string_column_mapping: Dict[str, set]) -> List[Tuple[str, str]]:

        def _find_ngrams(input_list, n):
            return zip(*[input_list[i:] for i in range(n)])

        def _lemmatizer_words(input_list):
            return [(Token(text=self.lemmatizer.lemmatize(word.text)), ) for word in input_list]

        entity_data = []
        ngram_indices = []
        ngram = [5, 4, 3, 2, 1, 0]
        for n in ngram:
            ngram_list = _find_ngrams(self.tokenized_utterance, n)
            if n == 0:
                ngram_list = _lemmatizer_words(self.tokenized_utterance)
                n = 1
            for i, tup in enumerate(ngram_list):
                # Check if any part of current ngram has already identified by other ngram.
                is_checked = False
                for ii in range(i, i + n):
                    if ii in ngram_indices:
                        is_checked = True
                        break
                if is_checked:
                    continue
                # Using token text to match schema entities
                tokens_text = [c.text for c in tup]
                if any(t in tokens_text for t in STOP_WORDS):
                    continue
                # Hard-code fix incorrect tokenized float number into three parts (e.g. ['1', '.', '84'])
                if len(tokens_text) == 3 and tokens_text[1] == '.' and is_number(tokens_text[0]) and is_number(
                        tokens_text[2]):
                    tokens_text = [''.join(tokens_text)]
                tokens = '_'.join(tokens_text)
                if not is_number(tokens):
                    normalized_token_text = self.normalize_string(tokens)
                else:
                    normalized_token_text = tokens
                # if is_number(normalized_token_text.replace('_', "")):
                #     normalized_token_text = normalized_token_text.replace('_', "")
                token_columns = self._string_in_table(
                    normalized_token_text, string_column_mapping
                )
                if token_columns:
                    # if n > 4: print(f"ngram:{n}")
                    normalized_token_text = normalized_token_text.replace('_', ' ')
                    token_type = ';'.join(list({token_column.split(":")[1] for token_column in token_columns}))
                    entity_data.append({'value': normalized_token_text,
                                        'token_start': i,
                                        'token_end': i + n,
                                        'token_type': token_type,
                                        'token_in_columns': token_columns})
                    ngram_indices.extend(range(i, i + n))
                elif is_number(normalized_token_text):
                    token_type = 'float' if str(normalized_token_text).count('.') == 1 else 'int'
                    entity_data.append({'value': normalized_token_text,
                                        'token_start': i,
                                        'token_end': i + n,
                                        'token_type': token_type,
                                        'token_in_columns': []})
                    ngram_indices.extend(range(i, i + n))
                else:
                    continue

        # extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = [(f"{entity['token_type']}:{entity['value']}", entity['token_in_columns']) for entity in
                             entity_data]
        # for entity in self._expand_entities(self.tokenized_utterance, entity_data, string_column_mapping):
        #     expanded_entities.append()
        # return expanded_entities, extracted_numbers  #TODO(shikhar) Handle conjunctions

        return expanded_entities

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = string.replace("_##", "")
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())

    def _expand_entities(self, question, entity_data, string_column_mapping: Dict[str, set]):
        new_entities = []
        for entity in entity_data:
            # to ensure the same strings are not used over and over
            if new_entities and entity['token_end'] <= new_entities[-1]['token_end']:
                continue
            current_start = entity['token_start']
            current_end = entity['token_end']
            current_token = entity['value']
            current_token_type = entity['token_type']
            current_token_columns = entity['token_in_columns']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self.normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" % (current_token, next_token_normalized)
                candidate_columns = self._string_in_table(candidate, string_column_mapping)
                candidate_columns = list(set(candidate_columns).intersection(current_token_columns))
                if not candidate_columns:
                    break
                candidate_type = candidate_columns[0].split(":")[1]
                if candidate_type != current_token_type:
                    break
                current_end += 1
                current_token = candidate
                current_token_columns = candidate_columns

            new_entities.append({'token_start': current_start,
                                 'token_end': current_end,
                                 'value': current_token,
                                 'token_type': current_token_type,
                                 'token_in_columns': current_token_columns})
        return new_entities
