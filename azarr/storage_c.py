import asyncio

import aiohttp
import requests
from zarr.storage import BaseStore


class SyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix
        self.session = requests.Session()

    def __getitem__(self, key):
        url = "/".join([self.prefix, key])
        try:
            return self.session.get(url).read()
        except Exception:
            pass
        return KeyError

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None


class ASyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix
        self.session = aiohttp.ClientSession()

    async def getitems(self, keys):
        urls = ["/".join([self.prefix, k]) for k in keys]
        responses = await asyncio.gather([self.session.get(url) for url in urls])
        out = await asyncio.gather([r.bytes() for r in responses if r.ok])
        valid_keys = [k for k, r in zip(keys, responses) if r.ok]
        return {k: o for k, o in zip(valid_keys, out)}

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None
