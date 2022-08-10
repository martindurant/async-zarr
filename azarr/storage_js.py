import asyncio

import pyodide.http

try:
    from zarr.storage import BaseStore
except ImportError:
    BaseStore = object


class SyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix

    def __getitem__(self, key):
        url = "/".join([self.prefix, key])
        try:
            return pyodide.http.open_url(url).read()
        except Exception as e:
            print(e)
        raise KeyError

    __delitem__ = __iter__ = __len__ = __setitem__ = None


class ASyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix

    async def getitems(self, keys):
        urls = ["/".join([self.prefix, k]) for k in keys]
        data = await asyncio.gather([get(url) for url in urls])
        return {k: o for k, o in zip(keys, data) if o}

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None


async def get(url):
    try:
        r = await pyodide.http.pyfetch(url)
        if r.ok:
            return await r.bytes()
    except Exception as e:
        print(e)
        return None
