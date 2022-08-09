try:
    from .storage_js import SyncStore, ASyncStore
except ImportError:
    from .storage_c import SyncStore, ASyncStore


__all__ = ["SyncStore", "ASyncStore"]
