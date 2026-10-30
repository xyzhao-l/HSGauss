def wu_palmer(predicted, expected, paths):
    if predicted == expected:
        return 1.0
    (predicted_path, expected_path) = (paths[predicted], paths[expected])
    (depth_pred, depth_true) = (len(predicted_path) - 1, len(expected_path) - 1)
    positions = {node: depth_pred - index for (index, node) in enumerate(predicted_path)}
    lca_depth = max((positions[node] for node in expected_path if node in positions), default=0)
    denominator = depth_pred + depth_true
    return 2.0 * lca_depth / denominator if denominator else 0.0

class RankingMetrics:

    def __init__(self, candidates, multi_parent, paths=None):
        self.candidates = candidates
        self.multi_parent = multi_parent
        self.paths = paths or {}
        self.queries = self.parent_count = 0
        self.rank_sum = self.reciprocal_sum = self.wu_sum = 0.0
        self.hits = {k: 0 for k in (1, 5, 10)}
        self.recalled = {k: 0 for k in (1, 5, 10)}

    def update(self, ranked_indices, ground_truth):
        predicted = [self.candidates[index] for index in ranked_indices]
        ranks = {node: index + 1 for (index, node) in enumerate(predicted)}
        expected = set(ground_truth)
        if not expected or not expected <= ranks.keys():
            raise ValueError('Missing ground-truth candidate')
        self.queries += 1
        self.parent_count += len(expected)
        self.rank_sum += sum((ranks[node] for node in expected))
        self.reciprocal_sum += sum((1.0 / ranks[node] for node in expected))
        for k in self.hits:
            count = len(set(predicted[:k]) & expected)
            self.hits[k] += int(count > 0)
            self.recalled[k] += count
        if not self.multi_parent:
            self.wu_sum += wu_palmer(predicted[0], next(iter(expected)), self.paths)

    def result(self):
        if not self.queries:
            raise ValueError('No evaluation queries')
        result = {'MR': self.rank_sum / self.parent_count, 'MRR': self.reciprocal_sum / self.parent_count}
        for k in self.hits:
            hit = self.hits[k] / self.queries
            result[f'Hit@{k}'] = hit
            result[f'Recall@{k}'] = self.recalled[k] / self.parent_count
            result[f'Prec@{k}'] = hit if self.multi_parent else hit / k
        if not self.multi_parent:
            result['WuP'] = result['Wu'] = self.wu_sum / self.queries
        return result
