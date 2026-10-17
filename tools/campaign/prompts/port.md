## Phase: port

Port the retained kernels to the QuixiCore library at `{{PORT_REPO}}`
per its sync contract (`csrc/quixicore/<platform>/README.md` in this
repo: same paths, same bytes, the README's build recipe). Kernels were
developed here first because that library connects to nothing; SlimServe
is where they are wired.

1. Read the sync contract and the last port entry in the status log.
2. Copy the retained kernel files (only the ones on the serving path) to
   the same paths in the library on a branch named after this campaign;
   build the library with its own recipe; run its tests.
3. Drift check: the kernel files must be byte-identical between the two
   trees after the port. Record the pair of commits.
4. Push the branch and open a draft PR there with a description that
   points at the SlimServe PR for the numbers. Same identity and text rules.

Marker evidence keys: `port_branch`, `port_pr` (if opened), `files` (count),
`drift` (`identical`).
