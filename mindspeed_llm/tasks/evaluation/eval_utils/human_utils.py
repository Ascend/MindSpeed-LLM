import re
import io
import os
import sys
import resource
import multiprocessing
import tempfile
import platform
import shutil
import signal
import json
import gzip
import subprocess
import builtins
import contextlib
import faulthandler
from typing import Optional, Callable, Dict, List, Iterable, Union

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import os.path as osp

import itertools

import numpy as np
import tqdm


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):

    def read(self, *args, **kwargs):
        """Disabled read operation."""
        raise IOError

    def readable(self, *args, **kwargs):
        """Always returns False indicating this stream is not readable."""
        return False

    def readlines(self, *args, **kwargs):
        """Disabled readlines operation."""
        raise IOError

    def readline(self, *args, **kwargs):
        """Disabled readline operation."""
        raise IOError


def humaneval_postprocess(text: str) -> str:
    blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]
    return text


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """Evaluates code correctness by executing it in a sandboxed environment.
    
    Args:
        problem: Dictionary containing coding problem details
        completion: Code solution to be tested
        timeout: Maximum execution time in seconds
        completion_id: Optional identifier for tracking
        
    Returns:
        Dictionary containing test results with keys:
        - task_id: Problem identifier
        - passed: Boolean indicating success
        - result: Detailed outcome string
        - completion_id: Echoed identifier
    """

    def unsafe_execute():
        """Runs untrusted code in a restricted environment with safety measures."""
        # Create a temporary directory for safe execution
        with create_tempdir():
            # Preserve original functions before modifying environment
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            os_chdir = os.chdir 
            # Apply security restrictions to prevent dangerous operations
            reliability_guard()

            # Construct full test program by combining:
            # 1. Problem prompt (function definitions)
            # 2. User's code completion
            # 3. Test cases
            # 4. Entry point verification call
            check_program = (
                problem["prompt"] + completion + "\n" +
                problem["test"] + "\n" + f"check({problem['entry_point']})"
            )
            
            # Global namespace for execution sandbox
            exec_globals = {}
            try:
                # Execute program with I/O suppressed and time limit
                with swallow_io():  # Redirects stdout/stderr
                    with time_limit(timeout):  # Enforces timeout
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:  # Catch any error during execution
                result.append(f"failed: {e}")
            # Restore original functions after execution
            shutil.rmtree = rmtree
            os.chdir = os_chdir
            os.rmdir = rmdir

    # Use multiprocessing for isolation and timeout enforcement
    manager = multiprocessing.Manager()
    result = manager.list()  # Shared list for inter-process communication

    # Execute untrusted code in separate process
    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    # Allow extra second beyond solution timeout for clean shutdown
    p.join(timeout=timeout + 1)
    
    # Terminate process if it exceeds allowed time
    if p.is_alive():
        p.kill()

    # Handle case where process was terminated before storing result
    if not result:
        result.append("timed out")

    # Format results with problem metadata
    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",  # Boolean success indicator
        result=result[0],  # Detailed status message
        completion_id=completion_id,  # Optional tracking ID
    )


@contextlib.contextmanager
def time_limit(seconds):
    """Context manager that enforces a time limit for code execution.
    
    Args:
        seconds: Maximum allowed execution time (in seconds)
        
    Raises:
        TimeoutException: If the code block exceeds the time limit
        
    Example:
        with time_limit(5):
            long_running_function()  # Will be interrupted after 5 seconds
    """
    # Signal handler function that gets called when the timer expires
    def signal_handler(signum, frame):
        """Handles SIGALRM signal by raising a timeout exception."""
        raise TimeoutException("Timed out!")

    # Set a one-shot timer that will send SIGALRM after specified seconds
    # ITIMER_REAL: Real-world timer (counts wall-clock time)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    # Register our handler for the alarm signal (SIGALRM)
    # Replaces any existing handler for this signal
    signal.signal(signal.SIGALRM, signal_handler)

    try:
        # Context manager entry point - executes the block inside "with"
        yield
    finally:
        # Cleanup: Always runs after the block completes or when interrupted
        # Cancel any pending alarm by setting timer to zero
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    """Context manager that silences all standard I/O streams (stdout, stderr, stdin).
    
    This effectively:
    - Redirects stdout/stderr to a void (discards all output)
    - Blocks input operations (stdin becomes write-only)
    
    Typical use case: Preventing untrusted code from flooding output or requesting input.
    """
    # Create a write-only buffer that discards output and blocks input
    stream = WriteOnlyStringIO()  # Custom class that disables read operations
    
    # Triple redirection stack:
    # 1. Redirect stdout to our write-only buffer
    with contextlib.redirect_stdout(stream):
        # 2. Redirect stderr to the same buffer
        with contextlib.redirect_stderr(stream):
            # 3. Redirect stdin to the same buffer (using custom redirect_stdin)
            with redirect_stdin(stream):
                # Execute the wrapped code block
                yield


class redirect_stdin(contextlib._RedirectStream):
    _stream = 'stdin'


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    """Context manager for temporarily changing the current working directory.
    
    Args:
        root: Target directory path (if ".", no change occurs)
        
    Behavior:
        - Changes to specified directory on entry
        - Restores original directory on exit
        - Handles exceptions safely
        - No-op if root is current directory (".")
    """
    # Short-circuit for current directory - no action needed
    if root == ".":
        yield  # Execute context without changing directory
        return  # Exit early
    
    # Preserve original working directory
    cwd = os.getcwd()
    # Change to target directory
    os.chdir(root)
    try:
        # Execute the wrapped code block
        yield
    except BaseException as exc:
        # Re-raise any exceptions that occur in the context
        # (Note: This except block is technically redundant since the exception 
        #  would propagate automatically, but shown for clarity)
        raise exc
    finally:
        # Critical cleanup: Always return to original directory
        os.chdir(cwd)


def write_jsonl(filename, data, append=False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def get_score(predictions, references, test_set, problem_set):
    prompts = [item['prompt'] for item in test_set]
    humaneval_preds = []
    for preds, refer in zip(predictions, references):
        if not isinstance(preds, list):
            preds = [preds]
        for pred in preds:
            humaneval_preds.append({'task_id': refer, 'completion': pred})
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = osp.join(tmp_dir, 'human_eval.json')
        write_jsonl(out_dir, humaneval_preds)
        score = evaluate_functional_correctness(sample_file=out_dir, problem_file=problem_set)

        detail_path = osp.join(tmp_dir, 'human_eval.json_results.jsonl')
        details = {}
        with open(detail_path, 'r') as f:
            for index, line in enumerate(f):
                line = json.loads(line)
                line['is_correct'] = line['passed']
                line['prompt'] = prompts[index]
                details[str(index)] = line

    results = {f'humaneval_{k}': score[k] * 100 for k in score}
    results['details'] = details
    return results


def evaluate_functional_correctness(
    sample_file: str,
    problem_file: dict = None,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    if problem_file is None:
        problem_file = {}

    problems = problem_file

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        if len(num_samples) != len(num_correct):
            raise ValueError(f"len(num_samples) must equal to len(num_correct)...")
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Hardens the execution environment against untrusted code.
    
    Applies multiple security layers:
    1. Memory limits
    2. Fault handler disabling
    3. Dangerous function neutralization
    4. Environment restrictions
    5. Module blocking
    
    Args:
        maximum_memory_bytes: Cap for memory usage (None = no limit)
    """
    
    # 1. MEMORY LIMITS - Prevent memory exhaustion attacks
    if maximum_memory_bytes is not None:
        # Set address space limit (virtual memory)
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        # Set data segment limit (heap memory)
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        
        # Stack limit (skip on macOS due to different behavior)
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    # 2. FAULT HANDLER - Disable crash dumps to prevent leaks
    faulthandler.disable()

    # 3. DANGEROUS BUILTINS - Neutralize exit mechanisms
    builtins.exit = None
    builtins.quit = None

    # 4. ENVIRONMENT RESTRICTIONS - Limit parallel processing
    os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP threads

    # 5. OS OPERATIONS - Disable dangerous system functions
    # Process control
    os.kill = None
    os.putenv = None
    os.system = None
    os.removedirs = None
    os.remove = None
    os.fchdir = None
    os.rmdir = None
    os.fork = None
    os.setuid = None
    os.killpg = None
    os.forkpty = None
    os.renames = None
    os.rename = None
    os.replace = None
    os.truncate = None
    os.fchmod = None
    os.unlink = None
    os.chmod = None
    os.fchown = None
    os.chroot = None
    os.chown = None
    os.lchflags = None
    os.fchdir = None
    os.lchown = None
    os.chdir = None
    os.getcwd = None
    os.lchmod = None

    shutil.rmtree = None
    shutil.chown = None
    shutil.move = None

    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
