from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def find(self, obj):
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj

        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objs):
        roots = [self.find(x) for x in objs]
        heaviest = max(roots, key=lambda r: self.weights[r])

        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest
                del self.weights[r]

        return heaviest

    def union_pairs(self, pair_gen):
        for x, y in pair_gen:
            self.union(x, y)

    def component_dict(self):
        result = defaultdict(list)
        for k in self.parents.keys():
            result[self.find(k)].append(k)
        return result
