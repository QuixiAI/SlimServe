@AGENTS.md

# Fork-specific rules (these override AGENTS.md)

This is a private fork specialized to serve GLM-5.2-Vision GGUF on MI300X. It
is not a source of upstream contributions, so the upstream contribution policy
in AGENTS.md does not apply here.

## Commit attribution: never credit an AI

Commits are authored by the human alone. **Do not add `Co-authored-by:`,
`Co-Authored-By:`, `Generated-by:` or `Assisted-by:` trailers naming Claude,
Anthropic, or any other AI tool**, and do not mention AI assistance in the
commit subject or body. This overrides the "Commit messages" section of
AGENTS.md, which asks for exactly such trailers — that guidance is upstream
vLLM's, and it does not apply to this repository.

The same applies to anything else that carries authorship: PR descriptions,
changelog entries, and file headers.

Note that upstream vLLM commits already in this history *do* carry AI
co-author trailers from their original authors. Leave them alone: they are
other people's authorship records, and rewriting them would also diverge this
fork's history from upstream for no benefit.
