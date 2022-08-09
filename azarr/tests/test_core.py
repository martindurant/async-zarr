import pytest

import azarr


@pytest.mark.asyncio
async def test1(server):
    g = azarr.open_group(server, mode="r")
    assert (await g.var[5:] == 1).all()
    assert (await g.var[:5] == 0).all()
