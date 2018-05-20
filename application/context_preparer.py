class ContextPreparer:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, item):
        return [self.normalizer(entry) for entry in item]


class FusedContextPreparer(ContextPreparer):
    def __call__(self, item):
        return [ContextPreparer.__call__(self, item)]
