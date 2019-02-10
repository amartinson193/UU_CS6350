
import numpy


def entropy(n,m):
    tot = n+m
    if n == 0 or m == 0:
        return 0
    return -(n/tot*numpy.log2(n/tot) + m/tot * numpy.log2(m/tot))


def gi(n,m):
    tot = n+m
    return 1-((n/tot)**2 + (m/tot)**2)

