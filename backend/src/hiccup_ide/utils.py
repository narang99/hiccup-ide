import itertools

def it_chain(iterator):
    return list(itertools.chain.from_iterable(iterator))