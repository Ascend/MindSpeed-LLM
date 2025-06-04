import time
import atexit
from multiprocessing.shared_memory import SharedMemory

import torch
import numpy as np
from megatron.core import mpu


class SharedMemoryManager:
    def __init__(self, base_shm_name, rank0_pid, buffer_length, tp_size,
                 existing=False, timeout=3600.0, sleep_time=0.001):
        """
        Achieve data sharing through the shared memory mechanism.
        :param base_shm_name: Base name for the shared memory (each TP group has independent shared memory when multiple TP groups exist)
        :param buffer_length: Size of the shared memory buffer (measured in int64 units)
        :param tp_size: TP group size (number of processes per TP group)
        :param existing: Whether to connect to existing shared memory
        :param timeout: Timeout duration for read/write operations (in seconds)
        :param sleep_time: Sleep duration during read/write waits (in seconds)
        """
        self.buffer_length = buffer_length
        self.tp_size = tp_size
        self.timeout = timeout
        self.sleep_time = sleep_time
        self.int64_size = torch.tensor(0, dtype=torch.int64).element_size()

        self.rank = mpu.get_tensor_model_parallel_rank()
        self.global_rank = torch.distributed.get_rank()
        self.tp_group_id = self.global_rank // self.tp_size

        if rank0_pid is None:
            raise ValueError("SharedMemoryManager requires rank0_pid to construct shm_name.")
        self.shm_name = self.generate_shm_name(base_shm_name, rank0_pid, self.tp_group_id)

        print(f"[SharedMemoryManager][Rank {self.rank}] <DEBUG> Using shm_name: {self.shm_name}")

        self.total_size = (buffer_length * self.int64_size + (tp_size + 3) * self.int64_size)

        if not existing:
            try:
                existing_shm = SharedMemory(name=self.shm_name)
                existing_shm.close()
                existing_shm.unlink()
                import multiprocessing.resource_tracker as rt
                rt.unregister(self.shm_name, "shared_memory")
                print(f"[SharedMemoryManager][Rank {self.rank}] <WARN> Unlinked residual shared memory '{self.shm_name}'.")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[SharedMemoryManager][Rank {self.rank}] <ERROR> Failed to unlink residual shared memory: {e}")

        self.shm = SharedMemory(
            name=self.shm_name,
            create=not existing,
            size=self.total_size if not existing else 0
        )

        offset = 0
        # Initial shared memory buffer
        self.tensor = np.frombuffer(self.shm.buf[offset:offset + buffer_length * self.int64_size],
                                       dtype=np.int64).reshape((buffer_length,))
        offset += buffer_length * self.int64_size

        self.seq_len_real_length = np.frombuffer(self.shm.buf[offset:offset + self.int64_size], dtype=np.int64)
        offset += self.int64_size

        self.seq_len_num = np.frombuffer(self.shm.buf[offset:offset + self.int64_size], dtype=np.int64)
        offset += self.int64_size

        self.read_flags = np.frombuffer(self.shm.buf[offset:offset + tp_size * self.int64_size], dtype=np.int64)
        offset += tp_size * self.int64_size

        self.data_version = np.frombuffer(self.shm.buf[offset:], dtype=np.int64)

        if not existing:
            self.read_flags.fill(0)
            self.data_version.fill(0)
            self.seq_len_real_length.fill(0)
            self.seq_len_num.fill(0)

        self.local_version = self.data_version.item()

        # Register a mechanism to automatically destroy shared memory
        atexit.register(self.close)

    @staticmethod
    def generate_shm_name(base_name, rank0_pid, tp_group_id):
        return f"{base_name}_pid{rank0_pid}_tp{tp_group_id}"

    def write(self, data):
        """
        Write data to the initialized shared memory buffer.

        Args:
            data: Data to be transferred via shared memory

        Returns:
            None
        """
        if self.rank != 0 or self.tp_size == 1:
            self.read_flags[self.rank] = 1
            return

        start_time = time.time()
        last_log_time = start_time
        # Waiting for all data to be read Translate English
        while self.data_version.item() > 0 and self.read_flags.sum().item() < self.tp_size:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                print(
                    f"[SharedMemoryManager][Rank {self.rank}]"
                    f"[global_rank {self.global_rank}][Func: write] <ERROR> "
                    f"Timeout: other ranks did not read data in time. "
                    f"read_flags: {self.read_flags.tolist()}"
                )
                self.read_flags[self.rank] = 1
                return

            if elapsed_time - last_log_time > 60.0:
                print(
                    f"[SharedMemoryManager][Rank {self.rank}]"
                    f"[global_rank {self.global_rank}][Func: write] <DEBUG> Waiting... "
                    f"Elapsed: {elapsed_time:.2f}s, "
                    f"read_flags sum = {self.read_flags.sum().item()} / {self.tp_size}"
                )
                last_log_time = time.time()
            time.sleep(self.sleep_time)

        if isinstance(data, list):
            if isinstance(data[0], torch.Tensor):
                data = [item.numpy() for item in data]
            data = torch.tensor(data, dtype=torch.int64)

        real_length = data.numel() if data is not None else 0
        seq_len_num = data.shape[0] if data is not None else 0
        self.read_flags.fill(0)

        if data is None or real_length == 0:
            print(
                f"[SharedMemoryManager][Rank {self.rank}]"
                f"[global_rank {self.global_rank}][Func: write] <WARN> "
                f"Writing None, setting seq_len_real_length=-1"
            )
            self.seq_len_real_length.fill(-1)
            self.seq_len_num.fill(-1)
        else:
            data_view_slice = data.view(-1)[:real_length]
            np.copyto(self.tensor[:real_length], data_view_slice.numpy())
            self.tensor[real_length:].fill(0)
            self.seq_len_real_length.fill(real_length)
            self.seq_len_num.fill(seq_len_num)

        np.add(self.data_version, 1, out=self.data_version)
        self.read_flags[self.rank] = 1

    def read(self):
        """
        Read data from the shared memory buffer.

        Returns:
            The retrieved data from shared memory
        """
        if self.rank == 0 or self.tp_size == 1:
            return None

        start_time = time.time()
        last_log_time = start_time
        # Waiting for data to be written
        while self.data_version.item() <= self.local_version:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                print(
                    f"[SharedMemoryManager][Rank {self.rank}]"
                    f"[global_rank {self.global_rank}][Func: read] <WARN> Timeout: No new data. "
                    f"data_version={self.data_version.item()}, "
                    f"local_version={self.local_version}"
                )
                return None

            if time.time() - last_log_time > 60.0:
                print(
                    f"[SharedMemoryManager][Rank {self.rank}]"
                    f"[global_rank {self.global_rank}][Func: read] <DEBUG> Still waiting... "
                    f"Elapsed: {elapsed_time:.2f}s, "
                    f"data_version={self.data_version.item()}, "
                    f"expected version > {self.local_version}"
                )
                last_log_time = time.time()
            time.sleep(self.sleep_time)
        # Get data from shared memory
        real_length = self.seq_len_real_length.item()
        seq_len_num = self.seq_len_num.item()
        # Invalid data
        if real_length == -1:
            print(
                f"[SharedMemoryManager][Rank {self.rank}]"
                f"[global_rank {self.global_rank}][Func: read] <INFO> "
                f"Detected None data (real_length=-1)"
            )
            data = None
        else:
            if seq_len_num <= 1:
                data_slice = self.tensor[:real_length].copy()
                data = torch.from_numpy(data_slice)
            else:
                data_slice = self.tensor[:real_length].copy()
                data_slice_reshape = data_slice.reshape(seq_len_num, -1)
                data = torch.from_numpy(data_slice_reshape)

        self.local_version = self.data_version.item()
        self.read_flags[self.rank] = 1

        if isinstance(data, torch.Tensor):
            data = data.tolist()
        return data

    def close(self):
        """
        Close handler registered with atexit for clean exit.

        Returns:
            None
        """
        # Wait for all data to be read
        if self.rank == 0:
            start_time = time.time()
            while self.read_flags.sum().item() < self.tp_size:
                if time.time() - start_time > self.timeout:
                    print(
                        f"[SharedMemoryManager][Rank {self.rank}]"
                        f"[global_rank {self.global_rank}][Func: close] <WARN> "
                        f"Timeout waiting for ranks to finish reading. "
                        f"read_flags: {self.read_flags.tolist()}"
                    )
                    break
                time.sleep(self.sleep_time)

        del self.tensor
        del self.seq_len_real_length
        del self.seq_len_num
        del self.read_flags
        del self.data_version

        import gc
        gc.collect()
        time.sleep(0.1)

        try:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
                print(
                    f"[SharedMemoryManager][Rank {self.rank}]"
                    f"[global_rank {self.global_rank}][Func: close] <INFO> "
                    f"Shared memory '{self.shm_name}' released and unlinked."
                )
            else:
                import multiprocessing.resource_tracker as rt
                rt.unregister(self.shm._name, "shared_memory")

        except Exception as e:
            print(
                f"[SharedMemoryManager][Rank {self.rank}]"
                f"[global_rank {self.global_rank}][Func: close] <ERROR> "
                f"Cleanup error during shm close/unlink: {e}"
            )
