import numpy as np
import zarr
from zarr.core import check_array_shape, check_fields, ensure_ndarray_like
from zarr.hierarchy import (
    ContainsArrayError,
    GroupNotFoundError,
    normalize_storage_path,
)
from zarr.convenience import PathNotFoundError
from zarr.storage import ConsolidatedMetadataStore, contains_array, contains_group


from .storage import SyncStore, ASyncStore


class AGroup(zarr.Group):
    def __getitem__(self, item):
        path = self._item_path(item)
        if contains_array(self._store, path):
            return AArray(
                self._store,
                read_only=self._read_only,
                path=path,
                chunk_store=self._chunk_store,
                synchronizer=self._synchronizer,
                cache_attrs=self.attrs.cache,
            )
        elif contains_group(self._store, path):
            return AGroup(
                self._store,
                read_only=self._read_only,
                path=path,
                chunk_store=self._chunk_store,
                cache_attrs=self.attrs.cache,
                synchronizer=self._synchronizer,
            )
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
            check_array_shape("out", out, out_shape)

        lchunk_coords, lchunk_selection, lout_selection = zip(*indexer)
        await self._chunk_getitems(
            lchunk_coords,
            lchunk_selection,
            out,
            lout_selection,
            drop_axes=indexer.drop_axes,
            fields=fields,
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _chunk_getitems(
        self,
        lchunk_coords,
        lchunk_selection,
        out,
        lout_selection,
        drop_axes=None,
        fields=None,
    ):
        """As _chunk_getitem, but for lists of chunks

        This gets called where the storage supports ``getitems``, so that
        it can decide how to fetch the keys, allowing concurrency.
        """
        out_is_ndarray = True
        try:
            out = ensure_ndarray_like(out)
        except TypeError:  # pragma: no cover
            out_is_ndarray = False

        ckeys = [self._chunk_key(ch) for ch in lchunk_coords]
        partial_read_decode = False
        cdatas = await self.chunk_store.getitems(ckeys, on_error="omit")
        for ckey, chunk_select, out_select in zip(
            ckeys, lchunk_selection, lout_selection
        ):
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


def open_group(
    store,
    mode="r",
    cache_attrs=True,
    synchronizer=None,
    path=None,
    chunk_store=None,
    **_,
):
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
    if isinstance(store, str):
        store = SyncStore(store)
    if chunk_store is None:
        chunk_store = ASyncStore(store.prefix)

    path = normalize_storage_path(path)

    if mode == "r":
        if not contains_group(store, path=path):
            if contains_array(store, path=path):
                raise ContainsArrayError(path)
            raise GroupNotFoundError(path)

    else:
        # although could have http PUT, leaving this example read-only for now
        raise NotImplementedError
    read_only = mode == "r"

    return AGroup(
        store,
        read_only=read_only,
        cache_attrs=cache_attrs,
        synchronizer=synchronizer,
        path=path,
        chunk_store=chunk_store,
    )


def open(store=None, mode="r", *, path=None, chunk_store=None, **kwargs):
    """Convenience function to open a group or array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or string, optional
        Store or path to directory in file system or name of zip file.
    mode : 'r', optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    path : str or None, optional
        The path within the store to open.
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    z : :class:`zarr.core.Array` or :class:`zarr.hierarchy.Group`
        Array or group, depending on what exists in the given store.

    See Also
    --------
    zarr.creation.open_array, zarr.hierarchy.open_group

    """

    if isinstance(store, str):
        store = SyncStore(store)
    if chunk_store is None:
        chunk_store = ASyncStore(store.prefix)

    path = normalize_storage_path(path)
    kwargs["path"] = path

    if contains_array(store, path):
        return AArray(store, read_only=True, chunk_store=chunk_store, **kwargs)
    elif contains_group(store, path):
        return open_group(store, mode=mode, chunk_store=chunk_store, **kwargs)
    else:
        raise PathNotFoundError(path)


def open_consolidated(store, metadata_key=".zmetadata", mode="r", **kwargs):
    """Open group using metadata previously consolidated into a single key.

    This is an optimised method for opening a Zarr group, where instead of
    traversing the group/array hierarchy by accessing the metadata keys at
    each level, a single key contains all of the metadata for everything.
    For remote data sources where the overhead of accessing a key is large
    compared to the time to read data.

    The group accessed must have already had its metadata consolidated into a
    single key using the function :func:`consolidate_metadata`.

    This optimised method only works in modes which do not change the
    metadata, although the data may still be written/updated.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    metadata_key : str
        Key to read the consolidated metadata from. The default (.zmetadata)
        corresponds to the default used by :func:`consolidate_metadata`.
    mode : {'r', 'r+'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist) although only writes to data are allowed,
        changes to metadata including creation of new arrays or group
        are not allowed.
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    g : :class:`zarr.hierarchy.Group`
        Group instance, opened with the consolidated metadata.

    See Also
    --------
    consolidate_metadata

    """

    # normalize parameters
    if isinstance(store, str):
        store = SyncStore(store)
    chunk_store = kwargs.pop("chunk_store", None)
    if chunk_store is None:
        chunk_store = ASyncStore(store.prefix)

    path = kwargs.pop("path", None)

    # setup metadata store
    meta_store = ConsolidatedMetadataStore(store, metadata_key=metadata_key)
    return open(
        store=meta_store, chunk_store=chunk_store, mode=mode, path=path, **kwargs
    )
