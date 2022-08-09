from zarr.storage import BaseStore


class FakeGetter(BaseStore):
    def __init__(self, *args, **kwargs):
        self.needed_keys = set()
        super().__init__(*args, **kwargs)

    async def getitems(self, items, **kwargs):
        self.needed_keys.update(items)
        return {}

    __delitem__ = __getitem__ = __iter__ = __len__ = __setitem__ = None
