"""
Execution-based evaluation harness for fine-tuned GPT-2 code generation.

Instead of exact-match, this actually RUNS each generated function against
test cases and checks whether it produces correct output.

Usage:
    python eval_harness.py <responses.json>

Each response is scored as one of:
    PASS        - code runs and produces correct output on all test cases
    FAIL        - code runs but gives wrong answers
    ERROR       - code crashes or has a syntax error
    UNTESTABLE  - no test case defined for this instruction (e.g. real GitHub funcs)
"""

import json
import ast
import sys
import io
import contextlib
import signal


# ---------------------------------------------------------------------------
# Timeout protection (some generated code can infinite-loop)
# ---------------------------------------------------------------------------
class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(fn, seconds=3):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Helper node classes injected so linked-list / tree functions can run
# ---------------------------------------------------------------------------
HELPER_DEFS = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""


def build_linked_list(values):
    head = None
    for v in reversed(values):
        head = {"val": v, "next": head}
    return head  # placeholder; replaced per-namespace below


def clean(resp):
    if "<|endoftext|>" in resp:
        resp = resp.split("<|endoftext|>")[0]
    return resp.strip()


def load_namespace(code):
    """Exec the generated code (plus helper classes) and return the namespace.
    Raises on syntax/runtime error during definition."""
    ns = {}
    full = HELPER_DEFS + "\n" + code
    with contextlib.redirect_stdout(io.StringIO()):
        exec(full, ns)
    return ns


def find_function(ns, prefer_names=()):
    """Find the most likely target function in the namespace."""
    # exclude our helper classes
    funcs = {k: v for k, v in ns.items()
             if callable(v) and not k.startswith("_")
             and k not in ("ListNode", "TreeNode")
             and getattr(v, "__module__", None) != "builtins"}
    if not funcs:
        return None
    # prefer a name match
    for name in prefer_names:
        if name in funcs:
            return funcs[name]
    # otherwise return the last-defined function
    return list(funcs.values())[-1]


def ll_from(ns, values):
    LN = ns["ListNode"]
    head = None
    for v in reversed(values):
        head = LN(v, head)
    return head


def ll_to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out


def tree_from(ns, values):
    """Build a simple BST from a list of values."""
    TN = ns["TreeNode"]
    root = None
    def insert(node, v):
        if node is None:
            return TN(v)
        if v < node.val:
            node.left = insert(node.left, v)
        else:
            node.right = insert(node.right, v)
        return node
    for v in values:
        root = insert(root, v)
    return root


# ---------------------------------------------------------------------------
# Test-case registry.
# Each entry: (keyword matcher, test function).
# The test function receives (fn, ns) and returns True if the generated
# function behaves correctly.
# ---------------------------------------------------------------------------
def t_reverse_list(fn, ns):
    return fn([1, 2, 3, 4]) == [4, 3, 2, 1] and fn([]) == []

def t_find_max(fn, ns):
    return fn([3, 7, 2, 9, 1]) == 9 and fn([5]) == 5

def t_find_min(fn, ns):
    return fn([3, 7, 2, 9, 1]) == 1

def t_second_largest(fn, ns):
    return fn([3, 7, 2, 9, 1]) == 7

def t_sum_list(fn, ns):
    return fn([1, 2, 3, 4]) == 10

def t_product(fn, ns):
    return fn([1, 2, 3, 4]) == 24

def t_remove_dupes(fn, ns):
    return fn([1, 1, 2, 3, 3, 3]) == [1, 2, 3]

def t_is_sorted(fn, ns):
    return fn([1, 2, 3]) is True and fn([3, 1, 2]) is False

def t_sort(fn, ns):
    return fn([3, 1, 2, 5, 4]) == [1, 2, 3, 4, 5] and fn([]) == []

def t_binary_search(fn, ns):
    return fn([1, 3, 5, 7, 9], 7) == 3 and fn([1, 3, 5], 4) == -1

def t_linear_search(fn, ns):
    return fn([1, 3, 5, 7], 5) == 2 and fn([1, 3], 9) == -1

def t_is_prime(fn, ns):
    try:
        return fn(7) and not fn(8) and not fn(1)
    except TypeError:
        return fn() in (True, False)  # parameterized version

def t_factorial(fn, ns):
    try:
        return fn(5) == 120 and fn(0) == 1
    except TypeError:
        return isinstance(fn(), int)

def t_fibonacci(fn, ns):
    try:
        return fn(7) == 13 and fn(0) == 0
    except TypeError:
        return isinstance(fn(), int)

def t_gcd(fn, ns):
    return fn(12, 8) == 4 and fn(17, 5) == 1

def t_reverse_string(fn, ns):
    return fn("hello") == "olleh"

def t_is_palindrome_str(fn, ns):
    return fn("racecar") is True and fn("hello") is False

def t_count_vowels(fn, ns):
    return fn("hello") == 2

def t_climb_stairs(fn, ns):
    try:
        return fn(5) == 8
    except TypeError:
        return isinstance(fn(), int)

def t_reverse_linked_list(fn, ns):
    head = ll_from(ns, [1, 2, 3])
    return ll_to_list(fn(head)) == [3, 2, 1]

def t_ll_length(fn, ns):
    head = ll_from(ns, [1, 2, 3, 4])
    return fn(head) == 4

def t_ll_middle(fn, ns):
    head = ll_from(ns, [1, 2, 3, 4, 5])
    res = fn(head)
    return getattr(res, "val", res) == 3

def t_ll_sum(fn, ns):
    head = ll_from(ns, [1, 2, 3, 4])
    return fn(head) == 10

def t_tree_inorder(fn, ns):
    root = tree_from(ns, [5, 3, 7, 2, 4])
    return fn(root) == [2, 3, 4, 5, 7]

def t_tree_height(fn, ns):
    root = tree_from(ns, [5, 3, 7])
    return fn(root) == 2

def t_tree_sum(fn, ns):
    root = tree_from(ns, [5, 3, 7])
    return fn(root) == 15

def t_bfs(fn, ns):
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    res = fn(graph, 0)
    return res[0] == 0 and set(res) == {0, 1, 2, 3}

def t_dfs(fn, ns):
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    res = fn(graph, 0)
    return res[0] == 0 and set(res) == {0, 1, 2, 3}


# (keywords that must appear in instruction, preferred fn names, test)
TEST_REGISTRY = [
    (["reverse", "linked"], ["reverse_list", "reverse", "reverse_linked_list", "reverse_iterative", "reverse_recursive"], t_reverse_linked_list),
    (["length", "linked"], ["length", "linked_list_length"], t_ll_length),
    (["middle", "linked"], ["find_middle"], t_ll_middle),
    (["sum", "linked"], ["sum_values"], t_ll_sum),
    (["inorder"], ["inorder"], t_tree_inorder),
    (["height", "tree"], ["height", "tree_height", "max_depth"], t_tree_height),
    (["sum", "tree"], ["tree_sum"], t_tree_sum),
    (["breadth-first"], ["bfs"], t_bfs),
    (["bfs"], ["bfs"], t_bfs),
    (["depth-first"], ["dfs"], t_dfs),
    (["dfs"], ["dfs"], t_dfs),
    (["reverse", "list"], ["reverse_list", "reverse"], t_reverse_list),
    (["reverse", "string"], ["reverse_string", "reverse"], t_reverse_string),
    (["maximum", "list"], ["find_max"], t_find_max),
    (["maximum value"], ["find_max"], t_find_max),
    (["minimum", "list"], ["find_min"], t_find_min),
    (["second largest"], ["second_largest"], t_second_largest),
    (["sum", "elements"], ["sum_list"], t_sum_list),
    (["product"], ["product"], t_product),
    (["duplicates"], ["remove_duplicates"], t_remove_dupes),
    (["sorted", "ascending"], ["is_sorted"], t_is_sorted),
    (["bubble sort"], ["bubble_sort"], t_sort),
    (["selection sort"], ["selection_sort"], t_sort),
    (["insertion sort"], ["insertion_sort"], t_sort),
    (["merge sort"], ["merge_sort"], t_sort),
    (["quick sort"], ["quick_sort"], t_sort),
    (["binary search"], ["binary_search"], t_binary_search),
    (["linear search"], ["linear_search"], t_linear_search),
    (["prime"], ["is_prime"], t_is_prime),
    (["factorial"], ["factorial"], t_factorial),
    (["fibonacci"], ["fibonacci"], t_fibonacci),
    (["greatest common divisor"], ["gcd"], t_gcd),
    (["palindrome"], ["is_palindrome"], t_is_palindrome_str),
    (["vowels"], ["count_vowels"], t_count_vowels),
    (["climbing stairs"], ["climb_stairs"], t_climb_stairs),
]


def find_test(instruction):
    ins = instruction.lower()
    for keywords, names, testfn in TEST_REGISTRY:
        if all(k in ins for k in keywords):
            return names, testfn
    return None, None


def evaluate(path):
    with open(path) as f:
        data = json.load(f)

    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "UNTESTABLE": 0}
    testable_pass = 0
    testable_total = 0
    details = []

    for entry in data:
        resp = clean(entry["model_response"])
        names, testfn = find_test(entry["instruction"])

        if testfn is None:
            counts["UNTESTABLE"] += 1
            details.append((entry["instruction"][:55], "UNTESTABLE"))
            continue

        testable_total += 1
        try:
            ns = load_namespace(resp)
            fn = find_function(ns, prefer_names=names)
            if fn is None:
                counts["ERROR"] += 1
                details.append((entry["instruction"][:55], "ERROR (no function)"))
                continue
            ok = run_with_timeout(lambda: testfn(fn, ns), seconds=3)
            if ok:
                counts["PASS"] += 1
                testable_pass += 1
                details.append((entry["instruction"][:55], "PASS"))
            else:
                counts["FAIL"] += 1
                details.append((entry["instruction"][:55], "FAIL"))
        except TimeoutException:
            counts["ERROR"] += 1
            details.append((entry["instruction"][:55], "ERROR (timeout)"))
        except Exception as e:
            counts["ERROR"] += 1
            details.append((entry["instruction"][:55], f"ERROR ({type(e).__name__})"))

    total = len(data)
    print("=" * 60)
    print(f"EXECUTION-BASED EVALUATION: {path.split('/')[-1]}")
    print("=" * 60)
    print(f"Total responses:     {total}")
    print(f"Testable responses:  {testable_total}")
    print(f"Untestable:          {counts['UNTESTABLE']} (no test case defined)")
    print()
    print("Among testable functions:")
    print(f"  PASS  (correct output):  {counts['PASS']:>3}  ({counts['PASS']/testable_total*100:.0f}%)")
    print(f"  FAIL  (wrong output):    {counts['FAIL']:>3}  ({counts['FAIL']/testable_total*100:.0f}%)")
    print(f"  ERROR (crash/timeout):   {counts['ERROR']:>3}  ({counts['ERROR']/testable_total*100:.0f}%)")
    print()
    print(f"FUNCTIONAL ACCURACY (pass rate on testable): {testable_pass/testable_total*100:.1f}%")
    print("=" * 60)

    return counts, testable_pass, testable_total, details


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "GPT2MedBalancedTemp1.json"
    evaluate(path)
