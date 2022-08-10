import asyncio

import aiohttp
import requests

try:
    from zarr.storage import BaseStore
except ImportError:
    BaseStore = object


class SyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix
        self.session = requests.Session()

    def __getitem__(self, key):
        url = "/".join([self.prefix, key])
        try:
            r = self.session.get(url)
            if r.ok:
                out = r.content
                return out
        except Exception:
            pass
        raise KeyError

    __delitem__ = __iter__ = __len__ = __setitem__ = None


class ASyncStore(BaseStore):
    def __init__(self, prefix):
        self.prefix = prefix
        self.session = aiohttp.ClientSession()

    async def getitems(self, keys, **_):
        urls = ["/".join([self.prefix, k]) for k in keys]
        data = await asyncio.gather(*[get(self.session, url) for url in urls])
        out = {k: o for k, o in zip(keys, data) if o}
        return out

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None

    def __del__(self):
        self.session.connector._close()
        self.session = None


async def get(session, url):
    try:
        r = await session.get(url)
        if r.ok:
            return await r.read()
    except Exception:
        return None
    finally:
        if "r" in locals():
            r.release()
