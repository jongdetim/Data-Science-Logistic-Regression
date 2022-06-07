class Descriptives:

    @staticmethod
    def sum(lst):
        res: float = 0
        for elem in lst:
            res += elem
            print(elem)
            print(res)
        return res
