import asyncio

import js.http
from zarr.storage import BaseStore


class SyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix

    def __getitem__(self, key):
        url = "/".join([self.prefix, key])
        try:
            return js.http.open_url(url).read()
        except Exception:
            pass
        raise KeyError

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None


class ASyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix

    async def getitems(self, keys):
        urls = ["/".join([self.prefix, k]) for k in keys]
        responses = await asyncio.gather([js.http.pyfetch(url) for url in urls])
        out = await asyncio.gather([r.bytes() for r in responses if r.ok])
        valid_keys = [k for k, r in zip(keys, responses) if r.ok]
        return {k: o for k, o in zip(valid_keys, out)}

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None
