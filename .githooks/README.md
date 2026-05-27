# .githooks

Repository git hooks (versioned in-tree so contributors share the same enforcement).

## Activate

```bash
git config core.hooksPath .githooks
```

`core.hooksPath` is a per-repo setting; the activation line above must be run once after each clone.

## Hooks

### `pre-commit`

Blocks `git commit` when there are untracked `.ts` / `.tsx` / `.svelte` / `.py` / `.mjs` / `.json` files under tracked directories (`viewer/src`, `viewer/scripts`, `src/indra_belief`, `tests`, `scripts`). Several cycles in this repo shipped broken on clean checkouts because new files were added locally but never `git add`-ed; the hook catches the class.

Bypass (use sparingly):

```bash
git commit --no-verify
```

If a file is intentionally untracked (a scratch / experimental path), add its pattern to `EXCLUDE_REGEX` inside the hook.
