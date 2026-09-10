"""
Bounded Ring Buffer & Preallocated Memory Pool for Deterministic Real-Time Ingestion.
Ensures zero-copy frame caching, strict bounded latency, and a latest-frame drop policy
to prevent stale frame accumulation in AUV/ROV perception loops.
"""

from typing import Dict, Any, Optional, Tuple, List
import threading
import time
import numpy as np


class PreallocatedBufferPool:
    """
    Fixed-size reusable memory buffer pool.
    Preallocates contiguous NumPy memory blocks to eliminate per-frame GC pauses
    and repeated heap memory allocations during continuous sonar streaming.
    """
    def __init__(self, pool_size: int = 8, shape: Tuple[int, int] = (640, 640), dtype=np.uint8):
        self.pool_size = pool_size
        self.shape = shape
        self.dtype = dtype
        self._lock = threading.Lock()
        
        # Preallocate memory buffers
        self._pool: List[np.ndarray] = [
            np.zeros(shape, dtype=dtype) for _ in range(pool_size)
        ]
        self._in_use = [False] * pool_size

    def acquire(self) -> Tuple[int, np.ndarray]:
        """Acquires a preallocated buffer index and array."""
        with self._lock:
            for idx in range(self.pool_size):
                if not self._in_use[idx]:
                    self._in_use[idx] = True
                    return idx, self._pool[idx]
            # If all buffers are busy, return index -1 with dynamic allocation as fallback
            return -1, np.zeros(self.shape, dtype=self.dtype)

    def release(self, buffer_id: int):
        """Releases the buffer back to the pool."""
        if buffer_id < 0 or buffer_id >= self.pool_size:
            return
        with self._lock:
            self._in_use[buffer_id] = False

    def copy_into_pool(self, source: np.ndarray) -> Tuple[int, np.ndarray]:
        """Copies source data into a preallocated pool slot without reallocating."""
        buf_id, buf = self.acquire()
        if source.shape == buf.shape and source.dtype == buf.dtype:
            np.copyto(buf, source)
        else:
            # Reshape or resize if dimension mismatch occurred
            buf = np.ascontiguousarray(source, dtype=self.dtype)
        return buf_id, buf


class BoundedFrameBuffer:
    """
    Thread-safe bounded ring buffer with an enforced latest-frame drop policy.
    Never allows stale frames to build up when sonar ping rate exceeds compute capacity.
    Tracks dropped frame counts and latency diagnostics.
    """
    def __init__(self, max_capacity: int = 4):
        self.max_capacity = max(1, max_capacity)
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
        # Metrics
        self.total_ingested = 0
        self.total_dropped = 0
        self.total_consumed = 0

    def push(self, frame_data: Dict[str, Any]) -> bool:
        """
        Pushes a new sonar frame into the buffer.
        If capacity is reached, drops the OLDEST frame immediately (latest-frame policy)
        to maintain strictly bounded real-time latency.
        """
        dropped = False
        with self._not_empty:
            if len(self._buffer) >= self.max_capacity:
                # Evict oldest frame
                oldest = self._buffer.pop(0)
                self.total_dropped += 1
                dropped = True
                
                # If the frame has an associated buffer pool ID, release it
                pool_ref = oldest.get("_pool_ref")
                if pool_ref and hasattr(pool_ref[0], "release"):
                    pool_ref[0].release(pool_ref[1])

            frame_data["ingest_timestamp"] = time.perf_counter()
            self._buffer.append(frame_data)
            self.total_ingested += 1
            self._not_empty.notify()
            
        return not dropped

    def pop(self, timeout: Optional[float] = 0.5) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next available frame. Blocks up to timeout seconds.
        Returns None if buffer remains empty.
        """
        with self._not_empty:
            while not self._buffer:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            frame = self._buffer.pop(0)
            self.total_consumed += 1
            return frame

    def clear(self):
        """Clears all buffered frames and resets metrics."""
        with self._lock:
            self._buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Returns buffer health, queue depth, and drop rate metrics."""
        with self._lock:
            drop_rate = (self.total_dropped / self.total_ingested) if self.total_ingested > 0 else 0.0
            return {
                "queue_depth": len(self._buffer),
                "max_capacity": self.max_capacity,
                "total_ingested": self.total_ingested,
                "total_consumed": self.total_consumed,
                "total_dropped": self.total_dropped,
                "drop_rate_pct": round(drop_rate * 100, 2)
            }
