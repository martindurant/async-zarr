import asyncio

import numpy as np
import zarr
from zarr.storage import contains_array, contains_group, meta_root
from zarr.core import check_fields, check_array_shape, ensure_ndarray
from zarr.hierarchy import (
    _normalize_store_arg, ContainsArrayError, ContainsGroupError, DEFAULT_ZARR_VERSION,
    GroupNotFoundError, init_group, normalize_storage_path
)


class AGroup(zarr.Group):
    def __getitem__(self, item):
        path = self._item_path(item)
        if contains_array(self._store, path):
            return AArray(self._store, read_only=self._read_only, path=path,
                          chunk_store=self._chunk_store,
                          synchronizer=self._synchronizer, cache_attrs=self.attrs.cache,
                          zarr_version=self._version)
        elif contains_group(self._store, path, explicit_only=True):
            return AGroup(self._store, read_only=self._read_only, path=path,
                          chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                          synchronizer=self._synchronizer, zarr_version=self._version)
        elif self._version == 3:
            implicit_group = meta_root + path + '/'
            # non-empty folder in the metadata path implies an implicit group
            if self._store.list_prefix(implicit_group):
                return AGroup(self._store, read_only=self._read_only, path=path,
                              chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                              synchronizer=self._synchronizer, zarr_version=self._version)
            else:
                raise KeyError(item)
        else:
            raise KeyError(item)


class AArray(zarr.Array):

    async def _get_selection(self, indexer, out=None, fields=None):

        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be extracted. Each chunk is processed in turn, extracting the
        # necessary data and storing into the correct location in the output array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the number of iterations in the main for loop.

        # check fields are sensible
        out_dtype = check_fields(fields, self._dtype)

        # determine output shape
        out_shape = indexer.shape

        # setup output array
        if out is None:
            out = np.empty(out_shape, dtype=out_dtype, order=self._order)
        else:
            check_array_shape('out', out, out_shape)

        # iterate over chunks
        if not hasattr(self.chunk_store, "getitems") or \
                any(map(lambda x: x == 0, self.shape)):
            # sequentially get one key at a time from storage
            for chunk_coords, chunk_selection, out_selection in indexer:

                # load chunk selection into output array
                await self._chunk_getitem(chunk_coords, chunk_selection, out, out_selection,
                                          drop_axes=indexer.drop_axes, fields=fields)
        else:
            # allow storage to get multiple items at once
            lchunk_coords, lchunk_selection, lout_selection = zip(*indexer)
            await self._chunk_getitems(lchunk_coords, lchunk_selection, out, lout_selection,
                                       drop_axes=indexer.drop_axes, fields=fields)

        if out.shape:
            return out
        else:
            return out[()]

    async def _chunk_getitem(self, chunk_coords, chunk_selection, out, out_selection,
                       drop_axes=None, fields=None):
        """Obtain part or whole of a chunk.

        Parameters
        ----------
        chunk_coords : tuple of ints
            Indices of the chunk.
        chunk_selection : selection
            Location of region within the chunk to extract.
        out : ndarray
            Array to store result in.
        out_selection : selection
            Location of region within output array to store results in.
        drop_axes : tuple of ints
            Axes to squeeze out of the chunk.
        fields
            TODO

        """
        out_is_ndarray = True
        try:
            out = ensure_ndarray(out)
        except TypeError:
            out_is_ndarray = False

        assert len(chunk_coords) == len(self._cdata_shape)

        # obtain key for chunk
        ckey = self._chunk_key(chunk_coords)

        try:
            # obtain compressed data for chunk
            cdata = await self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            if self._fill_value is not None:
                if fields:
                    fill_value = self._fill_value[fields]
                else:
                    fill_value = self._fill_value
                out[out_selection] = fill_value

        else:
            self._process_chunk(out, cdata, chunk_selection, drop_axes,
                                out_is_ndarray, fields, out_selection)

    async def _chunk_getitems(self, lchunk_coords, lchunk_selection, out, lout_selection,
                              drop_axes=None, fields=None):
        """As _chunk_getitem, but for lists of chunks

        This gets called where the storage supports ``getitems``, so that
        it can decide how to fetch the keys, allowing concurrency.
        """
        out_is_ndarray = True
        try:
            out = ensure_ndarray(out)
        except TypeError:  # pragma: no cover
            out_is_ndarray = False

        ckeys = [self._chunk_key(ch) for ch in lchunk_coords]
        partial_read_decode = False
        cdatas = await self.chunk_store.getitems(ckeys, on_error="omit")
        for ckey, chunk_select, out_select in zip(ckeys, lchunk_selection, lout_selection):
            if ckey in cdatas:
                self._process_chunk(
                    out,
                    cdatas[ckey],
                    chunk_select,
                    drop_axes,
                    out_is_ndarray,
                    fields,
                    out_select,
                    partial_read_decode=partial_read_decode,
                )
            else:
                # check exception type
                if self._fill_value is not None:
                    if fields:
                        fill_value = self._fill_value[fields]
                    else:
                        fill_value = self._fill_value
                    out[out_select] = fill_value


def open_group(store=None, mode='a', cache_attrs=True, synchronizer=None, path=None,
               chunk_store=None, storage_options=None, *, zarr_version=None):
    """Open a group using file-mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : string, optional
        Group path within store.
    chunk_store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group('data/example.zarr', mode='w')
    >>> foo = root.create_group('foo')
    >>> bar = root.create_group('bar')
    >>> root
    <zarr.hierarchy.Group '/'>
    >>> root2 = zarr.open_group('data/example.zarr', mode='a')
    >>> root2
    <zarr.hierarchy.Group '/'>
    >>> root == root2
    True

    """

    # handle polymorphic store arg
    store = _normalize_store_arg(
        store, storage_options=storage_options, mode=mode,
        zarr_version=zarr_version)
    # TODO: check metadata store is *not* async
    chunk_store = FakeGetter()

    path = normalize_storage_path(path)

    # ensure store is initialized

    if mode == 'r':
        if not contains_group(store, path=path):
            if contains_array(store, path=path):
                raise ContainsArrayError(path)
            raise GroupNotFoundError(path)

    else:
        raise NotImplementedError
    read_only = mode == 'r'

    return AGroup(store, read_only=read_only, cache_attrs=cache_attrs,
                  synchronizer=synchronizer, path=path, chunk_store=chunk_store,
                  zarr_version=zarr_version)


async def key_error_maker():
    raise KeyError


class FakeGetter(dict):
    def __init__(self, *args, **kwargs):
        self.needed_keys = set()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        self.needed_keys.add(item)
        return key_error_maker()

    def getitems(self, items):
        self.needed_keys.update(items)
        return {}
