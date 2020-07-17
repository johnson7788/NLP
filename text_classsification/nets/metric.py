# -- encoding:utf-8 --

from collections import namedtuple


class Metrics(namedtuple('Metrics',
                         ['accuracy', 'recall', 'f1'])):
    pass
