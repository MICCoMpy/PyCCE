import copy
from collections import defaultdict
from collections.abc import MutableMapping
from pycce.constants import PI2
from pycce.sm import _smc
import numpy as np
from pycce.utilities import kroncommute
from scipy.sparse._sputils import isintlike

two_layer_dict = lambda: defaultdict(dict)


class InteractionMap(MutableMapping):
    """
    Dict-like object containing information about tensor interactions between two spins.

    Each key is a tuple of two spin indexes.

    Args:
        rows (array-like with shape (n, )):
            Indexes of the bath spins, appearing on the left in the pairwise interaction.
        columns (array-like with shape (n, )):
            Indexes of the bath spins, appearing on the right in the pairwise interaction.
        tensors (array-like with shape (n, 3, 3)):
            Tensors of pairwise interactions between two spins with the indexes in ``rows`` and ``columns``.
    Attributes:
        mapping (dict): Actual dictionary storing the data.
    """

    def __init__(self, rows=None, columns=None, tensors=None):
        self.mapping = dict()
        self._indexes = None
        self._data = None
        self._data_shape = None
        if (rows is not None) & (columns is not None) & (tensors is not None):
            self[rows, columns] = tensors
            self.__gen_indexes()
            self._data_shape = tensors.shape
            # if self._data_shape != (3,3):
            #   print("Warning imap shape is not (3,3)")

    @property
    def indexes(self):
        """
        ndarray with shape (n, 2): Array with the indexes of pairs of spins, for which the tensors are stored.
        """
        if self._indexes is None:
            self.__gen_indexes()
        return self._indexes

    @indexes.setter
    def indexes(self, newindexes):
        self._indexes = newindexes

    @property
    def data(self):
        if self._data is None:
            self.__gen_data()
        return self._data

    def __getitem__(self, key):
        a, b = _index(key)
        try:
            if a < b:
                return self.mapping[a, b]
            else:
                return self.mapping[b, a].T if (self.mapping[b, a]).shape == (3, 3) else kroncommute(self.mapping[b, a])
        except (TypeError, ValueError):
            try:
                vs = []
                for j, k in zip(a, b):
                    v = self[j, k]
                    vs.append(v)
                return np.asarray(vs)

            except TypeError as e:
                raise TypeError("invalid index format") from e

    def __delitem__(self, key):
        a, b = _index(key)

        if isintlike(a) and isintlike(b):
            if a < b:
                del self.mapping[a, b]
            else:
                del self.mapping[b, a]
        else:
            for j, k in zip(a, b):
                if j < k:
                    del self.mapping[j, k]
                else:
                    del self.mapping[k, j]

        self.indexes = None
        self._data = None

    def __setitem__(self, key, value):
        if value.size == 9:
            value = value.reshape(3, 3)
        a, b = _index(key)

        if isintlike(a) and isintlike(b):
            assert value.size >= 9
            if a < b:
                self.mapping[a, b] = value
            elif value.size == 9 and b < a:
                self.mapping[b, a] = value.T
            else:
                self.mapping[b, a] = kroncommute(value)

        else:
            try:
                if value.size >= 9:
                    for j, k in zip(a, b):
                        if j < k:
                            self.mapping[j, k] = value
                        elif value.size == 9 and j > k:
                            self.mapping[k, j] = value.T
                        else:
                            self.mapping[k, j] = kroncommute(value)

                else:
                    for j, k, v in zip(a, b, value):
                        if j < k:
                            self.mapping[j, k] = v
                        elif v.size == 9 and j > k:
                            self.mapping[k, j] = v.T
                        else:
                            self.mapping[k, j] = kroncommute(v)

            except TypeError as e:
                raise TypeError("invalid index format") from e
        self._data = None
        self.indexes = None

    def shift(self, start, inplace=True):
        """
        Add an offset ``start`` to the indexes. If ``inplace`` is False, returns the copy of InteractionMap.

        Args:
            start (int): Offset in indexes.
            inplace (bool): If True, makes changes inplace. Otherwise returns copy of the map.

        Returns:
            InteractionMap: Map with shifted indexes.
        """

        if inplace:
            imap = self
        else:
            imap = copy.deepcopy(self)

        for i, j in imap.indexes:
            imap.mapping[i + start, j + start] = imap.mapping.pop((i, j))

        imap.indexes = None
        imap._data = None
        return imap

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.indexes)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    def keys(self):
        return self.mapping.keys()

    def items(self):
        return self.mapping.items()

    def __gen_indexes(self):
        self.indexes = np.fromiter((ind for pair in self.mapping.keys() for ind in pair), dtype=np.int32).reshape(-1, 2)

    def __gen_data(self):
        length = self.indexes.shape[0]
        s = self._data_shape
        self._data = np.zeros((length, *s), dtype=np.float64)
        # self._data = np.zeros((length, 3, 3), dtype=np.float64)

        for j, pair in enumerate(self.indexes):
            self._data[j] = self[pair]

    def subspace(self, array):
        r"""
        Get new InteractionMap with indexes readressed from array. Within the subspace indexes are renumbered.

        Examples:

            The subspace of [3,4,7] indexes will contain InteractionMap only within [3,4,7] elements
            with new indexes [0, 1, 2].

                >>> import numpy as np
                >>> im = InteractionMap()
                >>> im[0, 3] = np.eye(3)
                >>> im[3, 7] = np.ones(3)
                >>> for k in im: print(k, '\n', im[k],)
                (0, 3)
                [[1. 0. 0.]
                 [0. 1. 0.]
                 [0. 0. 1.]]
                (3, 7)
                 [[1. 1. 1.]
                  [1. 1. 1.]
                  [1. 1. 1.]]
                >>> array = [3, 4, 7]
                >>> sim = im.subspace(array)
                >>> for k in sim: print(k, '\n', sim[k])
                (0, 2)
                [[1. 1. 1.]
                 [1. 1. 1.]
                 [1. 1. 1.]]


        Args:
            array (ndarray): Either bool array containing True for elements within the subspace
                or array of indexes presented in the subspace.

        Returns:
            InteractionMap: The map for the subspace.
        """

        array = np.asarray(array)

        if array.dtype == bool:
            ind = np.arange(array.size, dtype=np.int32)
            indexes = ind[array]
        else:
            indexes = array

        newindexes = np.arange(indexes.size, dtype=np.int32)

        where = np.isin(self.indexes[:, 0], indexes) & np.isin(self.indexes[:, 1], indexes)
        pairs = self.indexes[where]

        xind = indexes.argsort()
        ypos = np.searchsorted(indexes[xind], pairs.flatten())

        indices = xind[ypos]
        newpairs = newindexes[indices].reshape(-1, 2)

        newdict = {}

        for (oi, oj), (i, j) in zip(pairs, newpairs):
            if i < j:
                newdict[i, j] = self[oi, oj]
            else:
                newdict[j, i] = self[oj, oi]

        return InteractionMap.from_dict(newdict, presorted=True)

    @classmethod
    def from_dict(cls, dictionary, presorted=False):
        """
        Generate InteractionMap from the dictionary.

        Args:

            dictionary (dict): Dictionary with tensors.

            presorted (bool): If true, assumes that the keys in the dictionary were already presorted.

        Returns:

            InteractionMap: New instance generated from the dictionary.
        """
        obj = cls()
        if presorted:
            obj.mapping = dictionary
        else:
            for k in dictionary:
                obj[k] = dictionary[k]
        return obj

    def __add__(self, other):
        new_obj = InteractionMap()
        keys_1 = list(self.keys())
        keys_2 = list(other.keys())

        for k in {*keys_1, *keys_2}:

            if (k in keys_1) and (k in keys_2):
                assert (other[k] == self[k]).all(), f"Error, tensor {k} has different properties in provided mappings"
                new_obj[k] = self[k]

            elif k in keys_1:
                new_obj[k] = self[k]
            else:
                new_obj[k] = other[k]
        return new_obj


def _index(key):
    try:
        a, b = key
    except TypeError as e:
        raise TypeError("invalid index format") from e
    return a, b


class LindbladianMap(MutableMapping):
    """ """

    def __init__(
        self,
    ):
        self.mapping = defaultdict(list)
        self._indexes = defaultdict(list)

    def __getitem__(self, key):
        if isinstance(key, int):  # single spin index
            condition = True
            key = (key,)
        else:
            condition = (not isinstance(key, np.ndarray) and (list(key) == sorted(key))) or (
                isinstance(key, np.ndarray) and (np.sort(key) == key).all()
            )  # if key is presorted

        if condition:  # if key is presorted just return the item
            try:
                return self.mapping[tuple(key)]
            except KeyError:
                return self.__missing__(key)

        else:
            # if key is not presorted return dissipators in desired order
            # note this creates new list of dissipators, so changing in-place is impossible
            sorted_indexes = np.argsort(key)
            sorted_key = sorted(key)
            how_to_reshuffle = np.argsort(sorted_indexes)

            try:
                return [diss.reorder(how_to_reshuffle) for diss in self.mapping[tuple(sorted_key)]]
            except KeyError:
                return self.__missing__(sorted_key)

    def __delitem__(self, key):
        if (not isinstance(key, np.ndarray) and (key == sorted(key))) or (
            isinstance(key, np.ndarray) and (np.sort(key) == key).all()
        ):
            del self.mapping[tuple(key)]

        else:
            sorted_key = sorted(key)
            del self.mapping[tuple(sorted_key)]

    def __len__(self):
        return len(self.mapping.keys())

    def __missing__(self, key):
        self.indexes[len(key)].append(tuple(key))
        self.mapping[tuple(key)] = []

    def __setitem__(self, key, value):

        raise ValueError("Direct setting of elements is not supported. Use add_dissipator instead")
        # You don't want to use direct setting as each element of the mapping is a list by default,
        # and you want to make sure user doesn't mess it up

    def add_dissipator(self, key, value):

        if not isinstance(value, Dissipator):
            raise TypeError("Elements of this map have to be of pycce.Dissipator class")

        if isinstance(key, int):  # single spin dissipator the key is always sorted
            condition = True
            key = (key,)
        else:
            condition = (not isinstance(key, np.ndarray) and (key == sorted(key))) or (
                isinstance(key, np.ndarray) and (np.sort(key) == key).all()
            )  # check if key is sorted

        if condition:
            # if key is sorted add dissipator to mapping directly
            tuple_key = tuple(key)

            if tuple_key not in self.mapping:
                self.indexes[len(key)].append(tuple(key))
            self.mapping[tuple_key].append(value)

        else:
            # otherwise sort the key and add a shifted dissipator
            sorted_indexes = np.argsort(key)
            tuple_key = tuple(sorted(key))
            how_to_reshuffle = np.argsort(sorted_indexes)
            value = value.reorder(how_to_reshuffle)

            if tuple_key not in self.mapping:
                self.indexes[len(key)].append(tuple(tuple_key))

            self.mapping[tuple_key].append(value)

    def shift(self, start, inplace=True):
        """
        Add an offset ``start`` to the indexes. If ``inplace`` is False, returns the copy of InteractionMap.

        Args:
            start (int): Offset in indexes.
            inplace (bool): If True, makes changes inplace. Returns copy of the map otherwise.

        Returns:
            InteractionMap: Map with shifted indexes.
        """

        if inplace:
            linmap = self
        else:
            linmap = copy.deepcopy(self)

        for k in linmap.indexes:
            for i, cluster in enumerate(linmap.indexes[k]):
                shifted_cluster = tuple(i + start for i in cluster)
                linmap.indexes[k][i] = shifted_cluster
                linmap.mapping[shifted_cluster] = linmap.mapping.pop(cluster)

        return linmap

    def __iter__(self):
        return iter(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    def __add__(self, other):
        new_obj = LindbladianMap()
        keys_1 = list(self.keys())
        keys_2 = list(other.keys())

        for k in {*keys_1, *keys_2}:

            if (k in keys_1) and (k in keys_2):
                assert other[k] == self[k], f"Error, tensor {k} has different properties in provided mappings"
                new_obj.mapping[k] = self[k]

            elif k in keys_1:
                new_obj.mapping[k] = self[k]
            else:
                new_obj.mapping[k] = other[k]
        return new_obj

    @property
    def indexes(self):
        """
        dict: Dictionary with each key n corresponding to list with the indexes of n-clusters of spins,
              for which the tensors are stored.
        """
        return self._indexes

    def keys(self):
        return self.mapping.keys()

    def items(self):
        return self.mapping.items()

    def subspace(self, array):
        r""" """

        array = np.asarray(array)

        if array.dtype == bool:
            ind = np.arange(array.size, dtype=np.int32)
            indexes = ind[array]
        else:
            indexes = array

        newindexes = np.arange(indexes.size, dtype=np.int32)

        newmappable = self.__class__()
        xind = indexes.argsort()

        for k in self.indexes:
            clusters = np.array(self.indexes[k])
            where = np.all([np.isin(clusters[:, i], indexes) for i in range(clusters.shape[1])], axis=0)
            pairs = clusters[where]  # only pairs included inside cluster

            ypos = np.searchsorted(indexes[xind], pairs.flatten())

            indices = xind[ypos]
            newpairs = newindexes[indices].reshape(pairs.shape)

            for cl, newcl in zip(pairs, newpairs):
                newmappable.mapping[tuple(newcl)] = self[tuple(cl)]

        return newmappable


class DissipatorList:
    def __init__(self):
        raise NotImplementedError


class Dissipator(object):
    def __init__(self, size=None, left=None, right=None, rate=1, symmetric=False):
        if size is None:
            size = len(left) if (left is not None) else len(right) if (right is not None) else None
        if size is None:
            raise ValueError("Could not determine the size of the Dissipator")

        self._size = size

        self._left = None
        self._right = None

        self.left[:] = left
        if symmetric:
            if right is None:
                right = left
            elif np.any(right != left):
                raise ValueError("symmetric key is set but left differs from right")

        self.right[:] = right

        self.rate = rate

        self.center_left = {}  # dict with center operators of the dissipator (left)
        self.center_right = {}  # dict with center operators of the dissipator (right)
        #  Each key is an index of the center in the centerarray. Agnostic to the total size of the CenterArray
        #  TODO make a check if a single center and then simplify addition

        self.symmetric = symmetric  # flag should be  manually set atm. Just reduces number of matrix multiplications

    @property
    def size(self):
        return self._size

    @property
    def left(self):
        if self._left is None:
            self._left = np.zeros(self.size, dtype=object)
        return self._left

    @left.setter
    def left(self, item):
        self._left[:] = item
        if self.symmetric:
            self._right[:] = item

    @property
    def right(self):
        if self._right is None:
            self._right = np.zeros(self.size, dtype=object)
        return self._right

    @right.setter
    def right(self, item):
        self._right[:] = item
        if self.symmetric:
            self._left[:] = item

    def reorder(self, neworder):
        """
        Reorder operators according to the indexes in the neworder.
        Args:
            neworder (array-like with length Dissipator.size):

        Returns:

        """
        neworder = np.asarray(neworder)
        if len(neworder) != self.size:
            raise ValueError(f"New order has length {len(neworder)} while expected {self.size}")

        left = self.left[neworder]
        right = self.right[neworder]
        newdis = Dissipator(left=left, right=right, symmetric=self.symmetric, rate=self.rate)
        newdis.center_right = self.center_right
        newdis.center_right = self.center_right
        return newdis

    def add_center_jump_operator(
        self,
        key,
        index=None,
        rate=None,
        side="both",
        units="rad",
        square_root=False,
    ):

        if rate is not None:
            # in dissipator object rate is stored in kHz (compared to sqrt(kHz) in .so dictionary),
            # we might want to change that...
            if square_root:
                rate = rate**2
            if "rad" in units:
                rate = rate / PI2
            self.rate = rate

        if index is None:
            index = 0

        if side == "both" or side == 2:
            side = "both"
            self.center_left[index] = key
            self.center_right[index] = key

        elif side == "left" or side == 0:
            self.center_left[index] = key
        elif side == "right" or side == 1:
            self.center_right[index] = key

        else:
            raise ValueError("Unsupported side. Allowed values are 'left' or 'right'")

        if self.symmetric and side != "both":
            raise ValueError("Different values for left and right are not allowed when symmetric = True")

    def add_jump_operator(
        self,
        key,
        index,
        rate=None,
        side="both",
        units="rad",
        square_root=False,
    ):

        if rate is not None:
            # in dissipator object rate is stored in kHz (compared to sqrt(kHz) in .so dictionary),
            # we might want to change that...
            if square_root:
                rate = rate**2
            if "rad" in units:
                rate = rate / PI2
            self.rate = rate

        if side == "both" or side == 2:
            side = "both"
            self.left[index] = key
            self.right[index] = key

        elif side == "left" or side == 0:
            self.left[index] = key
        elif side == "right" or side == 1:
            self.right[index] = key

        else:
            raise ValueError("Unsupported side. Allowed values are 'left' or 'right'")

        if self.symmetric and side != "both":
            raise ValueError("Different values for left and right are not allowed when symmetric = True")

    def __eq__(self, other):
        check_0 = self.size == other.size
        check_1 = (self.left == other.left).all()
        check_2 = (self.right == other.right).all()
        return check_0 & check_1 & check_2

    def __setitem__(self, key, value):
        if len(value) == 2 and not isinstance(value, str):
            self.left[key] = value[0]
            self.right[key] = value[1]
        else:
            self.left[key] = value
            self.right[key] = value

    def __getitem__(self, item):
        return self.left[item], self.right[item]


def _key_into_operator(key, spin):
    sm = _smc[spin]

    if isinstance(key, str):

        separated = key.split("+")
        operator = 0
        for k in separated:
            current = None
            for sym in k:
                current = getattr(sm, sym) if current is None else np.matmul(current, getattr(sm, sym))
            operator = operator + current

        operator = operator
    else:
        operator = sm.stev(*key)
    return operator


def process_key_dissipator(key, spin):
    if isinstance(key, np.ndarray):
        operator = key
    else:
        operator = _key_into_operator(key, spin)
    return operator


def process_key_operator(key, rate, spin):
    r"""
    Process key of the .so or .h dictionaries of the SpinType
    Args:
        key (str or int or tuple): key of the dictionary. Can be either of the following:

            * Pair of integers defining the Sven operator.
            * String where each symbol corresponds to the spin matrix or operation between them.
              Allowed symbols: ``xyz+``. If there is nothing between symbols, assume multiplication of the operators.
              If there is a ``+`` symbol, assume summation between terms. For example, ``xx+z`` would correspond to
              the operator :math:`\hat S_x \hat S_x + \hat S_z`.
            * String equal to ``A``. Then assumes that the correct matrix form of the operator has been provided
              by the user.


        rate (float or ndarray with shape (n,n): value stored in the dictionary.
        spin (float): Total spin of the spin.

    Returns:
        ndarray with shape (n,n): Resulting operator.
    """
    if isinstance(key, str) and key.lower() == "a":
        return rate
    operator = _key_into_operator(key, spin)
    return operator * rate
