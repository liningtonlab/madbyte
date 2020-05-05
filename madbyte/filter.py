
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
