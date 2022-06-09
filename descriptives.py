# ! Only to be used as a module, as it redefines built-in functions !

def sum(data):
    res = 0
    for elem in data:
        res += elem
    return res

def count(data):
    return len(data)

def mean(data):
    return sum(data) / count(data)

def min(data):
    smallest = float("inf")
    for elem in data:
        if elem < smallest:
            smallest = elem
    return smallest

def max(data):
    largest = float("-inf")
    for elem in data:
        if elem > largest:
            largest = elem
    return largest

def var(data, df=1):
    length = count(data)
    if length < 2:
        return None
    average = mean(data)
    return sum((elem - average) ** 2 for elem in data) / (length - df)

def std(data):
    return var(data) ** 0.5

def percentile(data, prcnt, interpolate=True):
    n = count(data)
    h = (n - interpolate) * prcnt / 100
    k = int(h)
    data = sorted(data)
    lower = data[k]
    if not interpolate:
        return lower
    return lower + (h - k) * (data[k+1] - lower)
    