# SPDX-License-Identifier: Apache-2.0
"""Compare saved cuobjdump SASS, including BOTH encoded 64-bit instruction words.

Raw disassembly must come from the named frozen binaries. Keep all duplicate
function copies per architecture. Reject missing scheduling words rather than
silently comparing only the mnemonic/first word. No GPU work or disassembly is
launched here; the full original raw files are preserved and hashed.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from benchmarks.kernels.replay_glm53_indexer import sha

INSTRUCTION = re.compile(r"\s*/\*([0-9a-f]+)\*/\s*.*?;")
CONTROL = re.compile(r"\s*/\*\s*0x[0-9a-f]{16}\s*\*/\s*$")
ENCODING = re.compile(r"/\*\s*0x[0-9a-f]{16}\s*\*/\s*$")


def function_bodies(lines):
    """Yield (architecture + symbol, ordered PC->full encoded instruction)."""
    arch, name, body, pending = None, None, {}, None
    for line in lines:
        if pending is not None:
            assert CONTROL.fullmatch(line), f"missing control word after {hex(pending)}"
            body[pending] += "\n" + line.strip()
            pending = None
            continue
        if re.search(r"arch = sm_", line):
            arch = line.strip()
        if "Function :" in line:
            if name is not None:
                assert body, name
                yield name, body
            assert arch is not None
            name = arch + " " + line.split("Function :", 1)[1].strip()
            body = {}
        instruction = INSTRUCTION.match(line)
        if instruction:
            assert name is not None and ENCODING.search(line)
            pc = int(instruction[1], 16)
            assert pc not in body
            body[pc] = line.strip()
            pending = pc
    assert pending is None, "truncated instruction scheduling word"
    assert name is not None and body
    yield name, body


def body_sha(body):
    return hashlib.sha256("\n".join(body.values()).encode()).hexdigest()


def signatures(path):
    result = {}
    with path.open() as stream:
        for name, body in function_bodies(stream):
            result.setdefault(name, Counter())[body_sha(body)] += 1
    return result


def compare(before, candidate):
    common = sorted(before.keys() & candidate.keys())
    changed = [name for name in common if before[name] - candidate[name]]
    removed = sorted(before.keys() - candidate.keys())
    added = sorted(
        name for name in candidate if candidate[name] - before.get(name, Counter())
    )
    identical = sum(sum((before[name] & candidate[name]).values()) for name in common)
    return dict(
        before_functions=sum(sum(values.values()) for values in before.values()),
        candidate_functions=sum(sum(values.values()) for values in candidate.values()),
        identical_common_functions=identical,
        changed_common_functions=changed,
        removed_functions=removed,
        added_functions=added,
        bodies={
            name: dict(
                before_instances=dict(before.get(name, {})),
                candidate_instances=dict(candidate.get(name, {})),
            )
            for name in sorted(before.keys() | candidate.keys())
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-binary", required=True, type=Path)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--before-sass", required=True, type=Path)
    parser.add_argument("--candidate-sass", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(signatures(args.before_sass), signatures(args.candidate_sass))
    result.update(
        comparison="all PCs and BOTH encoding words; duplicate copies retained",
        before_binary=dict(
            path=str(args.before_binary), sha256=sha(args.before_binary)
        ),
        candidate_binary=dict(
            path=str(args.candidate_binary), sha256=sha(args.candidate_binary)
        ),
        before_sass=dict(path=str(args.before_sass), sha256=sha(args.before_sass)),
        candidate_sass=dict(
            path=str(args.candidate_sass), sha256=sha(args.candidate_sass)
        ),
        source_sha256=sha(Path(__file__)),
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bodies"}, indent=2))


if __name__ == "__main__":
    main()
