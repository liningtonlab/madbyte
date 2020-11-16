
class Filter(object):
    """Simple class for storing filters
    """

    def __init__(self, columns, values):
        """Create Filter object

        Lengths of lists must match
        
        Args:
            columns (list): List of strings defining column names
            values (list): List of ranges indexed according to columns
        """
        # Make sure len columns matches len values
        assert len(columns) == len(values)
        self._shape = None
        self._data = dict(zip(columns, values))
        self._columns = None

    @property
    def shape(self):
        # Memoize this easily
        if not self._shape:
            self._shape = "undefined"
            col_len = len(self._data)
            if col_len == 1:
                self._shape = "stripe"
            elif col_len == 2:
                self._shape = "block"
        return self._shape

    @property
    def columns(self):
        # Memoize this easily
        if not self._columns:
            self._columns = list(self._data.keys())
        return self._columns

    def col_min(self, colname):
        assert colname in self._data.keys()
        return self._data[colname][0]

    def col_max(self, colname):
        assert colname in self._data.keys()
        return self._data[colname][1]


# TODO: Joe, add solvents as list of solvent peak + residual water peak
SOLVENT_DICT = {
    "dmso": [
        Filter(["H_PPM"], [(2.48, 2.52)]), 
        Filter(["H_PPM"], [(3.28, 3.32)])
    ], # DMSO+rH2O
    "chcl3": [
        Filter(["H_PPM"], [(0.00, 0.00)]), 
        Filter(["H_PPM"], [(1.54, 1.58)])
    ], # CHLOROFORM+rH2O
}