import json
import math
import pickle
import random
from collections import deque
from pathlib import Path
MULTI_PARENT = {'mesh', 'semeval_food'}

def lines(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [line.strip() for line in handle if line.strip()]

class SeedGraph:

    def __init__(self, nodes, edges):
        self.nodes = set(nodes)
        self.parents = {node: set() for node in self.nodes}
        self.children = {node: set() for node in self.nodes}
        for (parent, child) in edges:
            if parent == child:
                raise ValueError('Seed taxonomy contains a self edge')
            self.nodes.update((parent, child))
            self.parents.setdefault(child, set()).add(parent)
            self.parents.setdefault(parent, set())
            self.children.setdefault(parent, set()).add(child)
            self.children.setdefault(child, set())
        indegree = {node: len(self.parents[node]) for node in self.nodes}
        ready = deque(sorted((node for (node, degree) in indegree.items() if degree == 0)))
        visited = 0
        while ready:
            node = ready.popleft()
            visited += 1
            for child in sorted(self.children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(self.nodes):
            raise ValueError('Seed taxonomy must be acyclic')
        self.context_seed = 20

    def context(self, node, query, max_parents=4, max_siblings=4):
        if node == query:
            return ([], [])
        parents = self.parents.get(node, set()) - {query}
        siblings = set()
        for parent in parents:
            siblings.update(self.children[parent])
        siblings.difference_update((node, query))

        def select(values, role, cap):
            ordered = sorted(values)
            random.Random((self.context_seed + 1) * 1000003 + (node + 1) * 97 + role).shuffle(ordered)
            return ordered[:cap]
        return (select(parents, 0, max_parents), select(siblings, 1, max_siblings))

    def ancestors(self, node, hops=3, excluded=None):
        excluded = set() if excluded is None else set(excluded)
        queue = deque([(node, 0)])
        seen = {node}
        result = []
        while queue:
            (current, distance) = queue.popleft()
            if distance >= hops:
                continue
            for parent in sorted(self.parents.get(current, set())):
                if parent in seen or parent in excluded:
                    continue
                seen.add(parent)
                result.append((parent, distance + 1))
                queue.append((parent, distance + 1))
        return result

    def path_to_root(self, node):
        path = [node]
        while self.parents.get(node):
            node = min(self.parents[node])
            path.append(node)
        return path

class Taxonomy:

    def __init__(self, data_dir, dataset):
        self.dataset = dataset
        self.multi_parent = dataset in MULTI_PARENT
        self.directory = Path(data_dir) / dataset
        self.skipped_queries = 0
        if self.multi_parent:
            self._read_multi()
        else:
            self._read_single()
        self.graph = SeedGraph(self.seed_nodes, self.graph_edges)
        self.overlap_queries = set(self.queries) & self.seed_nodes
        candidates = set(self.candidates)
        for (query, parents) in zip(self.queries, self.ground_truth):
            if not parents or not set(parents) <= candidates:
                raise ValueError(f'Unrankable ground truth for query {self.names[query]!r}')
        self.paths = {} if self.multi_parent else {node: self.graph.path_to_root(node) for node in self.candidates}

    def _index(self, names):
        self.names = sorted(set(names))
        self.name_to_id = {name: index for (index, name) in enumerate(self.names)}

    def _read_single(self):
        full_pairs = []
        for line in lines(self.directory / f'{self.dataset}_raw_en.taxo'):
            fields = line.split('\t')
            (child, parent) = fields[-2:] if len(fields) >= 3 else fields
            full_pairs.append((parent, child))
        self._index((name for pair in full_pairs for name in pair))
        edges = [tuple((self.name_to_id[name] for name in line.split('\t'))) for line in lines(self.directory / f'{self.dataset}_train.taxo')]
        self.train_edges = edges
        self.seed_nodes = {node for edge in edges for node in edge}
        self.candidates = sorted(self.seed_nodes)
        self.graph_edges = list(edges)
        if self.dataset == 'wordnet':
            if self.dataset not in self.name_to_id:
                self.name_to_id[self.dataset] = len(self.names)
                self.names.append(self.dataset)
            root = self.name_to_id[self.dataset]
            roots = self.seed_nodes - {child for (_, child) in edges}
            self.graph_edges.extend(((root, node) for node in sorted(roots) if node != root))
            self.seed_nodes.add(root)
        definitions = json.loads((self.directory / 'dic.json').read_text(encoding='utf-8'))
        definitions = {name.lower(): definition for (name, definition) in definitions.items()}
        self.definitions = []
        for name in self.names:
            definition = definitions.get(name.lower(), ['Supernode'] if name == 'wordnet' else [name])
            self.definitions.append(str(definition[0] if isinstance(definition, list) else definition))
        query_names = lines(self.directory / f'{self.dataset}_eval.terms')
        parent_names = lines(self.directory / f'{self.dataset}_eval.gt')
        if len(query_names) != len(parent_names):
            raise ValueError('Query and ground-truth lengths differ')
        self.queries = [self.name_to_id[name] for name in query_names]
        self.ground_truth = [[self.name_to_id[name]] for name in parent_names]

    def _read_multi(self):
        terms = dict((line.split('\t', 1) for line in lines(self.directory / f'{self.dataset}.terms')))

        def read_edges(path):
            return [tuple((terms[node] for node in line.split('\t'))) for line in lines(path)]
        full_edges = read_edges(self.directory / f'{self.dataset}.taxo')
        self._index((name for edge in full_edges for name in edge))
        self.train_edges = [tuple((self.name_to_id[name] for name in edge)) for edge in read_edges(self.directory / f'{self.dataset}_train.taxo')]
        self.graph_edges = list(self.train_edges)
        self.seed_nodes = {node for edge in self.train_edges for node in edge}
        definitions = json.loads((self.directory / 'defs.json').read_text(encoding='utf-8'))
        self.definitions = [str(definitions.get(name.lower(), name.lower())) for name in self.names]
        truth = json.loads((self.directory / 'test_taxo.json').read_text(encoding='utf-8'))
        (self.queries, self.ground_truth) = ([], [])
        for line in lines(self.directory / f'{self.dataset}_test.terms'):
            (_, name) = line.split('\t', 1)
            if name not in self.name_to_id:
                self.skipped_queries += 1
                continue
            self.queries.append(self.name_to_id[name])
            self.ground_truth.append([self.name_to_id[parent] for parent in truth[name]])
        self.candidates = sorted(set(range(len(self.names))) - set(self.queries))

    def training_triples(self, negatives, seed, exp_id=0):
        cache = self.directory / 'processed' / f'taxonomy_data_{exp_id}{negatives}_.pkl'
        if not cache.is_file():
            raise FileNotFoundError('Training cache missing')
        with cache.open('rb') as handle:
            saved = pickle.load(handle)
        old_names = saved['id2concept']
        old_to_new = {int(old_id): self.name_to_id[name] for (old_id, name) in old_names.items()}
        triples = [tuple((old_to_new[int(node)] for node in triple)) for triple in saved['train_child_parent_negative_parent_triple']]
        if not triples:
            raise ValueError('Training cache empty')
        positives = {(child, parent) for (parent, child) in self.train_edges}
        if any(((child, parent) not in positives for (child, parent, _) in triples)):
            raise ValueError('Cached positive pairs do not match the current seed taxonomy')
        self.cached_negative_conflicts = sum((negative == child or negative in self.graph.parents[child] for (child, _, negative) in triples))
        return triples

def build_context_batch(data, pairs, max_parents=4, max_siblings=4, enabled=True):
    pairs = list(dict.fromkeys(pairs))
    jobs = list(pairs)
    job_ids = {pair: index for (index, pair) in enumerate(jobs)}
    contexts = {}

    def get_context(pair):
        if pair not in contexts:
            (query, node) = pair
            contexts[pair] = data.graph.context(node, query, max_parents, max_siblings) if enabled else ([], [])
        return contexts[pair]
    for (query, anchor) in pairs:
        (parents, _) = get_context((query, anchor))
        for parent in parents:
            pair = (query, parent)
            if pair not in job_ids:
                job_ids[pair] = len(jobs)
                jobs.append(pair)
    for pair in jobs:
        get_context(pair)
    nodes = {node for (_, node) in jobs}
    nodes.update((query for (query, _) in pairs if query >= 0))
    nodes.update((n for (parents, siblings) in contexts.values() for n in parents + siblings))
    nodes = sorted(nodes)
    lookup = {node: index for (index, node) in enumerate(nodes)}
    return {'node_ids': nodes, 'pairs': pairs, 'jobs': jobs, 'job_nodes': [lookup[node] for (_, node) in jobs], 'parent_nodes': [[lookup[n] for n in contexts[pair][0]] for pair in jobs], 'sibling_nodes': [[lookup[n] for n in contexts[pair][1]] for pair in jobs], 'anchor_nodes': [lookup[node] for (_, node) in pairs], 'anchor_jobs': [job_ids[pair] for pair in pairs], 'anchor_parent_jobs': [[job_ids[query, p] for p in contexts[query, node][0]] for (query, node) in pairs]}

def perturb_siblings(layout, rng, drop_rate=0.25):
    if not 0 <= drop_rate < 1:
        raise ValueError('Sibling drop rate must be in [0, 1)')
    counterfactual = dict(layout)
    (siblings, changed_jobs) = ([], [])
    for row in layout['sibling_nodes']:
        count = min(len(row) - 1, math.ceil(len(row) * drop_rate)) if len(row) >= 2 else 0
        removed = set(rng.sample(range(len(row)), count)) if count else set()
        siblings.append([node for (index, node) in enumerate(row) if index not in removed])
        changed_jobs.append(bool(removed))
    counterfactual['sibling_nodes'] = siblings
    changed_anchors = [bool(layout['parent_nodes'][job]) and (changed_jobs[job] or any((changed_jobs[parent] for parent in parents))) for (job, parents) in zip(layout['anchor_jobs'], layout['anchor_parent_jobs'])]
    return (counterfactual, changed_anchors)

def build_batch(data, triples, max_parents=4, max_siblings=4, enabled=True, cf_drop_rate=0.0, cf_rng=None):
    pairs = [(q, candidate) for (q, p, n) in triples for candidate in (p, n)]
    batch = build_context_batch(data, pairs, max_parents, max_siblings, enabled)
    lookup = {node: index for (index, node) in enumerate(batch['node_ids'])}
    anchors = {pair: index for (index, pair) in enumerate(batch['pairs'])}
    batch.update(query_indices=[lookup[q] for (q, _, _) in triples], parent_indices=[anchors[q, p] for (q, p, _) in triples], negative_indices=[anchors[q, n] for (q, _, n) in triples])
    if enabled and cf_drop_rate > 0:
        if cf_rng is None:
            raise ValueError('Counterfactual batches require a dedicated random generator')
        (cf_layout, changed) = perturb_siblings(batch, cf_rng, cf_drop_rate)
        batch['cf_layout'] = cf_layout
        batch['cf_active'] = [(changed[p_index] or changed[n_index]) and n != q and (n not in data.graph.parents.get(q, set())) for ((q, _, n), p_index, n_index) in zip(triples, batch['parent_indices'], batch['negative_indices'])]
    return batch
