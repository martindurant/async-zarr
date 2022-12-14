import asyncio
import weakref

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
        self.session = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            weakref.finalize(self, self.close, self.session)
        return self.session

    async def getitems(self, keys, **_):
        urls = ["/".join([self.prefix, k]) for k in keys]
        session = await self.get_session()
        data = await asyncio.gather(*[get(session, url) for url in urls])
        out = {k: o for k, o in zip(keys, data) if o}
        return out

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None

    @staticmethod
    def close(session):
        connector = getattr(session, "_connector", None)
        if connector is not None:
            # close after loop is dead
            connector._close()


async def get(session, url):
    try:
        r = await session.get(url)
        if r.ok:
            return await r.read()
    except Exception as e:
        print(e)
        return None
    finally:
        if "r" in locals():
            await r.__aexit__(None, None, None)
